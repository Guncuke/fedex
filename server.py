import torch
from models import Model
from aggregation import *


class Server(Model):

	_instance = None

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			cls._instance = super(Server, cls).__new__(cls)
		return cls._instance

	def __init__(
        self,
        model_name: str,
        batch_size: int,
        eval_dataset,
        aggr_rule: str
    ):
		super().__init__(model_name, eval_dataset)
		self.aggr_rule = aggr_rule
		self.global_model = self.model
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

	

	def model_update(self, clients_diff, clients_data_len):

		weight_accumulator = self.simple_avg(clients_diff)

		if self.aggr_rule == "SimpleAbg":
			weight_accumulator = self.simple_avg(clients_diff)
		elif self.aggr_rule == "FedAvg":
			weight_accumulator = self.fed_avg(clients_diff, clients_data_len)
		elif self.aggr_rule == "MaxAvg":
			weight_accumulator = self.max_avg(clients_diff)
		else:
			pass

		for name, params in weight_accumulator.items():

			# Only use state_dict(), the model has Int type
			if self.global_model.state_dict()[name].dtype != weight_accumulator[name].dtype:
				self.global_model.state_dict()[name] += weight_accumulator[name].to(self.global_model.state_dict()[name].dtype)
			else:
				self.global_model.state_dict()[name] += weight_accumulator[name]

	def model_eval(self):
		self.global_model.eval()
		loss = 0.0
		correct = 0
		dataset_size = 0

		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			dataset_size += data.size()[0]

			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()

			output = self.global_model(data)
			loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
			pred = output.data.max(1)[1]
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		loss = loss / dataset_size
		return acc, loss
