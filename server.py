import torch
from models import Model


class Server(Model):
	
	def __init__(
		self,
		model_name: str,
		batch_size: int,
		eval_dataset
	):
		super().__init__(model_name, eval_dataset)
		self.global_model = self.model
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

	@staticmethod
	def model_aggregation(self, clients_diff):

		weight_accumulator = {}
		for name, params in clients_diff[0].items():
			weight_accumulator[name] = torch.zeros_like(params)

		for _, client_diff in enumerate(clients_diff):
			for name, params in client_diff.items():
				weight_accumulator[name].add_(params)

		for name, params in weight_accumulator.items():
			update_per_layer = params * len(clients_diff)

			if params.type() != update_per_layer.type():
				params.add_(update_per_layer.to(torch.int64))
			else:
				params.add_(update_per_layer)

	def model_eval(self):
		self.global_model.eval()
		total_loss = 0.0
		correct = 0
		dataset_size = 0

		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			output = self.global_model(data)
			total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
			pred = output.data.max(1)[1]
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size
		return acc, total_l
