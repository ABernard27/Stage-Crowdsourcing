from function import function
from gradient_ascent import gradient_ascent
from matplotlib import pyplot as plt


def plot_func(n_workers, n_items, n_classes, data, sigma, tau, Q, n_iter=10,
              learning_rate=0.01):
    """
    Plots the output of a function over a specified number of iterations.

    Parameters:
        n_workers (int): The number of workers.
        n_items (int): The number of items.
        n_classes (int): The number of classes.
        data (dict): The input data.
        sigma (torch.Tensor): The sigma parameter.
        tau (torch.Tensor): The tau parameter.
        Q (dict): The distribution Q.
        n_iter (int, optional): The number of iterations. Defaults to 10.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
    """
    x = list(range(n_iter))
    vec = list(range(n_iter))
    for i in range(n_iter):
        vec[i] = function(n_workers, n_items, n_classes, data, sigma, tau)
        sigma, tau = gradient_ascent(n_workers, n_classes, n_items, data,
                                     sigma, tau, Q, learning_rate,
                                     max_iterations=1)
    plt.plot(x, vec)
