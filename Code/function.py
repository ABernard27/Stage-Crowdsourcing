import torch
from annex_def import params, dist_Q
import numpy as np


def function(n_workers, n_items, n_classes, data, sigma, tau):
    """
    Calculates the value of the function based on the given parameters.

    Parameters:
        n_workers (int): The number of workers.
        n_items (int): The number of items.
        n_classes (int): The number of classes.
        data (dict): A dictionary containing the data.
        sigma (torch.Tensor): A tensor representing sigma.
        tau (torch.Tensor): A tensor representing tau.

    Returns:
        float: The calculated value of the function.
    """
    func = 0
    alpha, beta = params(data, n_classes)
    Q = dist_Q(data, n_classes)
    omega = 0
    psi = 0

    for w in range(n_workers):
        for i in range(n_items):
            for v in range(n_classes):
                for c in range(n_classes):
                    st = sigma[w]+tau[i]
                    k = data[f"{w}"][f"{i}"]
                    P = torch.log(np.exp(st))-torch.logsumexp(st, dim=0)
                    func += Q[f"{i}"][f"{v}"]*P[v, k]
                    omega += sigma[w, v, c]**2
                    psi += tau[i, v, c]**2
    func = func + alpha*(1/2)*omega + beta*(1/2)*psi
    return func.tolist()
