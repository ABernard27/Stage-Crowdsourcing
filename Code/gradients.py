from annex_def import params
import torch


def gradient_sigma(n_workers, n_classes, n_items, data,
                   sigma, tau, Q):
    """
    Computes the gradient of the sigma parameter in a multi-class \
    classification problem.

    Parameters:
        n_workers (int): The number of workers.
        n_classes (int): The number of classes.
        n_items (int): The number of items.
        data (dict): The data for the classification problem.
        sigma (torch.Tensor): The sigma parameter.

    Returns:
        DF_sigma (torch.Tensor): The gradient of the sigma parameter.
    """
    alpha = params(data, n_classes)[0]
    DF_sigma = torch.zeros((n_workers, n_classes, n_classes))

    for w in range(n_workers):
        for v in range(n_classes):
            for c in range(n_classes):
                DFc = 0
                ind = 0
                for i in range(n_items):
                    # expo = 0
                    if data[f"{w}"][f"{i}"] == c:
                        ind = 1
                    else:
                        ind = 0
                    k = data[f"{w}"][f"{i}"]
                    st = sigma[w]+tau[i]
                    P = torch.softmax(st, dim=1)
                    DFc += Q[f"{i}"][f"{v}"]*(ind - P[v, k])
                DF_sigma[w, v, c] = DFc-alpha*sigma[w, v, c]
    return DF_sigma


def gradient_tau(n_workers, n_classes, n_items, data,
                 sigma, tau, Q):
    """
    Computes the gradient of the tau parameter in a multi-class \
    classification problem.

    Parameters:
        n_workers (int): The number of workers.
        n_classes (int): The number of classes.
        n_items (int): The number of items.
        data (dict): The data for the classification problem.
        sigma (torch.Tensor): The sigma parameter.
        tau (torch.Tensor): The tau parameter.
        Q (matrix): Distribution of the labels.

    Returns:
        DF_tau (torch.Tensor): The gradient of the tau parameter.
    """
    beta = params(data, n_classes)[1]
    DF_tau = torch.zeros((n_items, n_classes, n_classes))

    for i in range(n_items):
        for v in range(n_classes):
            for c in range(n_classes):
                DFc = 0
                ind = 0
                for w in range(n_workers):
                    # expo = 0
                    if data[f"{w}"][f"{i}"] == c:
                        ind = 1
                    else:
                        ind = 0
                    k = data[f"{w}"][f"{i}"]
                    st = sigma[w]+tau[i]
                    P = torch.softmax(st, dim=1)
                    DFc += Q[f"{i}"][f"{v}"]*(ind - P[v, k])
                DF_tau[i, v, c] = DFc-beta*tau[i, v, c]

    return DF_tau
