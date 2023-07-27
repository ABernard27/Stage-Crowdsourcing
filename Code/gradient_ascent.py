from gradients import gradient_sigma, gradient_tau


def gradient_ascent(n_workers, n_classes, n_items, data, sigma, tau, Q,
                    learning_rate=0.01, max_iterations=10):
    """
    Perform gradient ascent optimization.

    Args:
        n_workers (int): Number of workers.
        n_classes (int): Number of classes.
        n_items (int): Number of items.
        data (dict): Data observed.
        sigma (torch.Tensor): Sigma parameter.
        learning_rate (float): Learning rate.
        max_iterations (int): Maximum number of iterations.

    Returns:
        sigma, tau (float): Sigma and tau values.
    """

    for _ in range(max_iterations):
        sigma_up = sigma + learning_rate * \
            gradient_sigma(n_workers, n_classes, n_items, data, sigma, tau, Q)
        tau_up = tau + learning_rate * \
            gradient_tau(n_workers, n_classes, n_items, data, sigma, tau, Q)
        sigma = sigma_up
        tau = tau_up
    return sigma_up, tau_up
