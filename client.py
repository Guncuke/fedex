import torch


class Client(object):

	client_id = 0

	def __init__(
		self,
		batch_size: int,
		lr: float,
		momentum: float,
		model_parameter: str,
		local_epochs: int,
		model,
		train_dataset,
	) -> None:

		Client.client_id += 1
		self.client_id = Client.client_id
		self.batch_size = batch_size
		self.lr = lr
		self.momentum = momentum
		self.model_parameter = model_parameter
		self.local_epochs = local_epochs
		self.local_model = model
		self.train_dataset = train_dataset
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

	def local_train(self):

		# record the previous model parameters
		# 1. calculate diff
		# 2. restoring the global model
		pre_model = {}
		if self.model_parameter == "all":
			for name, param in self.local_model.state_dict().items():
				pre_model[name] = param.clone()
		else:
			for name, param in self.local_model.named_parameters():
				pre_model[name] = param.clone()

		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr, momentum=self.momentum)

		epoch = self.local_epochs

		self.local_model.train()

		for _ in range(epoch):
			for _, batch in enumerate(self.train_loader):
				data, target = batch

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
				
		print(f"{self.client_id} complete!")

		# record the differences between the local model and the global model
		diff = {}

		for name, param in pre_model.items():
			diff[name] = self.local_model.state_dict()[name] - param

		for name, param in pre_model.items():
			self.local_model.state_dict()[name] = param

		return diff, len(self.train_dataset)
