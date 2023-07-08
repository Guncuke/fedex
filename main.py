import json
from torch.utils.data import Subset
from server import Server
from client import Client
import datasets
import random
import distribution


if __name__ == '__main__':

    # load the configure file
    with open('./conf.json', 'r') as f:
        conf = json.load(f)

    # load dataset
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["dataset"])

    server = Server(batch_size=conf["batch_size"],
                    lamda=conf["lambda"],
                    model_name=conf["model_name"],
                    eval_dataset=eval_datasets)

    # total clients array
    clients = []

    if conf["data_distribution"] == 'iid':
        n_clients = conf["num_models"]
        data_len = len(train_datasets)
        subset_indices = distribution.split_iid(n_clients, data_len)
        for idx in subset_indices:
            subset_dataset = Subset(train_datasets, idx)
            clients.append(Client(batch_size=conf["batch_size"],
                                  lr=conf["lr"],
                                  momentum=conf["momentum"],
                                  model_parameter=conf["model_parameter"],
                                  local_epochs=conf["local_epochs"],
                                  model=server.global_model,
                                  train_dataset=subset_dataset))

    elif conf["data_distribution"] == 'dirichlet':
        n_clients = conf["num_models"]
        dirichlet_alpha = conf["dirichlet_alpha"]
        train_labels = train_datasets.targets
        # return an array: every client's index
        client_idcs = distribution.dirichlet_split_noniid(train_labels, alpha=dirichlet_alpha, n_clients=n_clients)

        for c, subset_indices in enumerate(client_idcs):
            subset_dataset = Subset(train_datasets, subset_indices)
            clients.append(Client(batch_size=conf["batch_size"],
                                  lr=conf["lr"],
                                  momentum=conf["momentum"],
                                  model_parameter=conf["model_parameter"],
                                  local_epochs=conf["local_epochs"],
                                  model=server.global_model,
                                  train_dataset=subset_dataset))

    accuracy = []
    losses = []

    for e in range(conf["global_epochs"]):

        # random choice k clients
        candidates = random.sample(clients, conf["k"])

        # clients_weight recode the diffs of every client
        clients_weight = []

        for _, c in enumerate(candidates):
            diff = c.local_train()
            clients_weight.append(diff)

        server.model_aggregation(clients_diff=clients_weight)

        acc, loss = server.model_eval()
        accuracy.append(acc)
        losses.append(loss)

        print(f"Epoch {e:d}, acc: {acc:f}, loss: {loss:f}\n")
