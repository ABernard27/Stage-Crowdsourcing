# %%

import torch
import random
from annex_def import dist_Q
from annex_def import item_json
from gradient_ascent import gradient_ascent
from update_Q import update_Q
from plot_func import plot_func
from function import function
import json
import numpy as np


# %%

# code test à ne pas utiliser, voir les derniers

n_workers = 10
n_items = 10
n_classes = 5
# création des données
dic = {}
for w in range(n_workers):
    dic[f"{w}"] = {}
    for i in range(n_items):
        dic[f"{w}"][f"{i}"] = random.randint(0, n_classes-1)

dic1 = {'0': {'0': 2, '1': 4, '2': 0, '3': 3, '4': 1},
        '1': {'0': 2, '1': 4, '2': 0, '3': 3, '4': 1},
        '2': {'0': 2, '1': 4, '2': 0, '3': 3, '4': 1},
        '3': {'0': 2, '1': 4, '2': 0, '3': 3, '4': 1},
        '4': {'0': 2, '1': 4, '2': 0, '3': 3, '4': 1}}

dic2 = {'0': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1},
        '1': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1},
        '2': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1},
        '3': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1}}


sigma = torch.zeros((n_workers, n_classes, n_classes))
tau = torch.zeros((n_items, n_classes, n_classes))
sigma1 = torch.ones((n_workers, n_classes, n_classes))
tau1 = torch.ones((n_items, n_classes, n_classes))
sigma2 = torch.rand((n_workers, n_classes, n_classes))
tau2 = torch.rand((n_items, n_classes, n_classes))

num_iterations = 50
Q = dist_Q(dic, n_classes)

for _ in range(num_iterations):
    sigma_up, tau_up = gradient_ascent(
        n_workers, n_classes, n_items, dic, sigma, tau, Q)
    Q = update_Q(n_workers, n_classes, n_items, sigma_up, tau_up)
    sigma = sigma_up
    tau = tau_up

# %%

# plot de la fonction en fontion des initialisations et pas

# création des données
n_workers = 10
n_items = 10
n_classes = 5
dic = {}
for w in range(n_workers):
    dic[f"{w}"] = {}
    for i in range(n_items):
        dic[f"{w}"][f"{i}"] = random.randint(0, n_classes-1)
Q = dist_Q(dic, n_classes)

# initialisation des paramètres
sigma = torch.zeros((n_workers, n_classes, n_classes))
tau = torch.zeros((n_items, n_classes, n_classes))

sigma1 = torch.ones((n_workers, n_classes, n_classes))
tau1 = torch.ones((n_items, n_classes, n_classes))

sigma2 = torch.rand((n_workers, n_classes, n_classes))
tau2 = torch.rand((n_items, n_classes, n_classes))

value = function(n_workers, n_items, n_classes, dic, sigma, tau)
print(value)

# plot de la fonction
plot_func(n_workers, n_items, n_classes, dic, sigma, tau, Q,
          n_iter=50, learning_rate=0.01)


# %%

# importation données bluebirds
n_workers = 39
n_items = 108
n_classes = 2

json_file_path = "../Data/labels_bluebirds.json"
with open(json_file_path, "r") as json_file:
    bluebirds = json.load(json_file)

for key in bluebirds.keys():
    for value in bluebirds[key].keys():
        if bluebirds[key][value] is True:
            bluebirds[key][value] = 1
        elif bluebirds[key][value] is False:
            bluebirds[key][value] = 0

# Création des nouvelles clés
new_keys = []
for i in range(n_workers):
    new_keys.append(str(i))
key_mapping = dict(zip(bluebirds, new_keys))
bluebirds = {key_mapping[key]: value for key, value in bluebirds.items()}

new_keys2 = []
for i in range(n_items):
    new_keys2.append(str(i))
for j in range(n_workers):
    key_mapping2 = dict(zip(bluebirds[f"{j}"], new_keys2))
    bluebirds[f"{j}"] = {key_mapping2[key]: value for key, value
                         in bluebirds[f"{j}"].items()}

bluebirds_dataset = '../Data/bluebirds.json'
with open(bluebirds_dataset, 'w') as file:
    json.dump(bluebirds, file)

# test du code avec bluebirds
sigma = torch.zeros((n_workers, n_classes, n_classes))
tau = torch.zeros((n_items, n_classes, n_classes))
num_iterations = 10
Q = dist_Q(bluebirds, n_classes)

for _ in range(num_iterations):
    sigma_up, tau_up = gradient_ascent(
        n_workers, n_classes, n_items, bluebirds, sigma, tau, Q)
    Q = update_Q(bluebirds, n_workers, n_classes, n_items, sigma_up, tau_up)
    sigma = sigma_up
    tau = tau_up

with open('../output/result_bluebird.json', 'w') as fp:
    json.dump(Q, fp)

with open('../output/result_bluebird.json', 'r') as fp:
    data = json.load(fp)

# %%

# test avec hammer spammer peerannot

n_workers = 10
n_items = 100
n_classes = 5

path_hm = "../Data/simulation/answers.json"
with open(path_hm, "r") as json_file:
    hammer_spammer = json.load(json_file)

hammer_spammer = item_json(hammer_spammer)

sigma = torch.ones((n_workers, n_classes, n_classes))
tau = torch.ones((n_items, n_classes, n_classes))
num_iterations = 10
Q = dist_Q(hammer_spammer, n_classes)

for _ in range(num_iterations):
    sigma_up, tau_up = gradient_ascent(
        n_workers, n_classes, n_items, hammer_spammer, sigma, tau, Q)
    Q = update_Q(hammer_spammer, n_workers, n_classes, n_items, sigma_up,
                 tau_up)
    sigma = sigma_up
    tau = tau_up

with open('../output/result_hs.json', 'w') as fp:
    json.dump(Q, fp)

with open('../output/result_hs.json', 'r') as fp:
    data = json.load(fp)
# %%

# calcul de l'erreur pour bluebirds et hammer spammer
path1 = "../output/result_bluebird.json"
with open(path1, "r") as json_file:
    rbluebirds = json.load(json_file)
path2 = "../output/result_hs.json"
with open(path2, "r") as json_file:
    rhs = json.load(json_file)

# bluebirds
error_b = 0
for i in range(48):
    vec = []
    for valeur in rbluebirds[f"{i}"].values():
        vec.append(valeur)
    if max(vec) == vec[0]:
        error_b = error_b+1
for j in range(49, 108):
    vec = []
    for valeur in rbluebirds[f"{j}"].values():
        vec.append(valeur)
    if max(vec) == vec[1]:
        error_b = error_b+1

ERROR_B = error_b/len(rbluebirds)*100

# hammer spammer
error_hs = 0
result = np.load('../Data/simulation/ground_truth.npy')
vec = []
for i in range(len(rhs)):
    vec = []
    for valeur in rhs[f"{i}"].values():
        vec.append(valeur)
    if max(vec) != vec[result[i]]:
        error_hs = error_hs+1

ERROR_HS = error_hs/len(rhs)*100
# %%
