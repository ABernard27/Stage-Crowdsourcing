def count(list, target):
    """
    Counts the number of occurrences of a target value in a list.

    Parameters:
        list (list): The list to search for the target value.
        target: The value to count in the list.

    Returns:
        int: The number of occurrences of the target value in the list.
    """
    n = len(list)
    res = 0
    for i in range(n):
        if list[i] == target:
            res += 1
    return res


def get_vec(list, n_classes):
    """
    Calculates the vector representation of a list based on the number of\
        occurrences of each class.

    Parameters:
        list (list): The input list.
        n_classes (int): The number of classes.

    Returns:
        vec (list): The vector representation of the input list.
    """
    vec = []
    for i in range(n_classes):
        vec.append(count(list, i)/len(list))
    return vec


def worker_json(answers):
    """
    Generate a dictionary that organizes worker answers by task and worker.

    Parameters:
    - answers (dict): A dictionary containing worker answers for each task.

    Returns:
    - w_answer (dict): A dictionary that organizes worker answers by task\
          and worker.
    """
    w_answer = {}
    for task, dict in answers.items():
        for worker, classes in dict.items():
            if worker in w_answer:
                w_answer[worker][task] = classes
            else:
                w_answer[worker] = {}
                w_answer[worker][task] = classes
    return w_answer


def item_json(answers):
    """
    Generates a JSON object from a nested dictionary of answers.

    Parameters:
    - answers (dict): A nested dictionary containing worker IDs, task names,\
    and lists of classes.

    Returns:
    - w_answer (dict): A JSON object representing the answers,\
    with tasks as keys, and worker IDs as subkeys,\
    each containing a list of classes.
    """
    w_answer = {}
    for worker, dict in answers.items():
        for task, classes in dict.items():
            if task in w_answer:
                w_answer[task][worker] = classes
            else:
                w_answer[task] = {}
                w_answer[task][worker] = classes
    return w_answer


def dist_Q(data, n_classes):
    """
    Calculate the distribution of the labels given by the observation

    Parameters:
    - data (dict): A dict with worker's answers.
    - n_classes (int): The number of classes.

    Returns:
    - dict: A dictionary containing the distribution of Q
    """
    d = {}
    for i in range(len(worker_json(data))):
        vec = get_vec(list(worker_json(data)[f"{i}"].values()), n_classes)
        d[f"{i}"] = {}
        for c in range(n_classes):
            d[f"{i}"][f"{c}"] = vec[c]
    return d


def params(data, n_classes):
    """
    Calculates the values of `alpha` and `beta` based on the given `data`\
        and `n_classes`.

    Parameters:
        data (dict): A dictionary containing data.
        n_classes (int): The number of classes.

    Returns:
        tuple: A tuple containing the values of `alpha` and `beta`.
    """
    gamma = 1/2
    alpha = gamma*(n_classes)**2

    workerj = worker_json(data)
    lw = 0
    for i in range(len(workerj)):
        lw += len(workerj[f"{i}"])
    lwm = round(lw/len(workerj))
    li = 0
    for i in range(len(data)):
        li += len(data[f"{i}"])
    lim = round(li/len(data))

    beta = (lwm/lim)*alpha

    return alpha, beta
