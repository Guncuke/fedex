import torch
from torch.utils.data import Subset
from server import Server
from client import Client
import datasets
import random
from utils import distribution
from typing import Optional


class Controller(object):

    def __init__(self,
                 dataset,
                 batch_size: int,
                 model_name: str,
                 num_client: int,
                 data_distribution: str,
                 dirichlet_alpha: Optional[float],
                 lr: float,
                 momentum: float,
                 model_parameter: str,
                 local_epochs: int,
                 aggr_rule: str
                 ):

        train_datasets, eval_datasets = datasets.get_dataset("./data/", dataset)
        self.server = Server(batch_size=batch_size,
                             model_name=model_name,
                             eval_dataset=eval_datasets,
                             aggr_rule=aggr_rule)

        # total clients array
        self.clients = []

        if data_distribution == 'iid':
            subset_indices = distribution.split_iid(num_client, len(train_datasets))

        elif data_distribution == 'dirichlet':
            subset_indices = distribution.dirichlet_split_noniid(train_datasets.targets, alpha=dirichlet_alpha,
                                                                 n_clients=num_client)
        else:
            # write yourself func in distribution.py
            subset_indices = distribution.split_iid(num_client, len(train_datasets))

        # self.data_distribution : to draw the data distribution graph
        n_classes = train_datasets.targets.max()+1
        self.data_distribute = [[0]*n_classes for i in range(num_client)]
        for i, indices in enumerate(subset_indices):
            for j in indices:
                self.data_distribute[i][train_datasets.targets[j]] += 1

        for idx in subset_indices:
            subset_dataset = Subset(train_datasets, idx)
            self.clients.append(Client(batch_size=batch_size,
                                       lr=lr,
                                       momentum=momentum,
                                       model_parameter=model_parameter,
                                       local_epochs=local_epochs,
                                       model=self.server.global_model,
                                       train_dataset=subset_dataset))

        self.accuracy = []
        self.losses = []

    def run(self, k: int):

        # random choice k clients
        candidates = random.sample(self.clients, k)

        # clients_weight recode the diffs of every client
        clients_weight = []
        clients_data_len = []

        for _, c in enumerate(candidates):
            diff, data_len = c.local_train()
            clients_weight.append(diff)
            clients_data_len.append(data_len)

        self.server.model_update(clients_weight, clients_data_len)

        acc, loss = self.server.model_eval()
        self.accuracy.append(acc)
        self.losses.append(loss)

        print(f"acc: {acc:f}, loss: {loss:f}\n")
