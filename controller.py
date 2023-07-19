from torch.utils.data import Subset
from server import Server
from client import Client
import datasets
import random
import distribution
from typing import Optional
import streamlit as st


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
                 ):

        train_datasets, eval_datasets = datasets.get_dataset("./data/", dataset)
        self.server = Server(batch_size=batch_size,
                             model_name=model_name,
                             eval_dataset=eval_datasets)

        # total clients array
        self.clients = []

        n_clients = num_client

        if data_distribution == 'iid':
            subset_indices = distribution.split_iid(n_clients, len(train_datasets))

        elif data_distribution == 'dirichlet':
            dirichlet_alpha = dirichlet_alpha
            train_labels = train_datasets.targets
            subset_indices = distribution.dirichlet_split_noniid(train_labels, alpha=dirichlet_alpha,
                                                                 n_clients=n_clients)
        else:
            # write yourself func in distribution.py
            subset_indices = distribution.split_iid(n_clients, len(train_datasets))

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

        for _, c in enumerate(candidates):
            diff = c.local_train()
            clients_weight.append(diff)

        self.server.model_aggregation(clients_diff=clients_weight)

        acc, loss = self.server.model_eval()
        self.accuracy.append(acc)
        self.losses.append(loss)

        print(f"acc: {acc:f}, loss: {loss:f}\n")
