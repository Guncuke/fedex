import torch
from torch.distributions.dirichlet import Dirichlet


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))
    # 1. Get the index of each label
    class_idcs = [torch.nonzero(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 2. According to the distribution, the label is assigned to each client
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs


def split_iid(n_clients, data_len):
    client_data_len = data_len // n_clients
    clients_idx = [[y for y in range(x * client_data_len, (x + 1) * client_data_len)] for x in range(n_clients)]
    redundant = list(range(client_data_len*n_clients, data_len))
    for i, inx in enumerate(redundant):
        clients_idx[i] += [inx]

    return clients_idx

