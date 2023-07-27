import torch
import numpy as np


def update_Q(data, n_workers, n_classes, n_items, sigma, tau):
    """
    Update Q.

    Args:
        data (dict): Data for optimization.
        n_workers (int): Number of workers.
        n_classes (int): Number of classes.
        n_items (int): Number of items.
        sigma (tensor): Sigma tensor.
        tau (tensor): Tau tensor.
    """
    Q = {}
    for i in range(n_items):
        Q[f"{i}"] = {}
        Q_list = []
        for v in range(n_classes):
            S = 0
            for w in range(n_workers):
                k = data[f"{w}"][f"{i}"]
                st = sigma[w]+tau[i]
                sumexp = torch.logsumexp(st, dim=1)
                S = S + st[v, k].item() - sumexp[v].item()
            Q[f"{i}"][f"{v}"] = np.exp(S)
            Q_list.append(Q[f"{i}"][f"{v}"])
        for v1 in range(n_classes):
            Q[f"{i}"][f"{v1}"] = Q[f"{i}"][f"{v1}"]/np.sum(Q_list)
    return Q
