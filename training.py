# Performs a training of the current MNN model
# TODO: store the generated graphs and data to make sure the experiments are replicable
# TODO: store results of experiments in a file

import dgl
from generate_dataset import DatasetGenerator
import torch
import torch.nn as nn
from mpnn import MPNN
import time
import networkx as nx
import numpy as np

# Setting the seed for replicability
import random
random.seed(33)

#####################################################
# --- Training parameters
#####################################################

# For the training hyperparameters, insire from paper
nb_epochs = 10
nb_features = 32
lr = 0.0005

# Datasets parameters
graph_type = 'erdos_renyi'
nb_graphs = 200
nb_nodes = 20
algorithm_type = 'DFS'

max_steps = nb_nodes + 1 # maximum number of steps before stopping
# I added +1 as experimentally the case happends, t investigate

####################################################
# --- Data generation
####################################################

start = time.time()
data_gen = DatasetGenerator()
graphs, history_dataset = data_gen.run(graph_type, nb_graphs, nb_nodes,
                                algorithm_type)

print('Dataset created in:', time.time()-start)
clock = time.time()

# Inspect some shapes
for i in range(10):
    pass
    #print(history_dataset[i].shape)

# dataset is an array of size batch size
# each idem is a history of shape (nb_steps, nb_nodes)
# We now have in our possession a full dataset on which to train


###################################################

model = MPNN(1, 32, 1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

def nll_gaussian(preds, target):
    neg_log_p = ((preds - target) ** 2)
    return neg_log_p.sum() / (target.size(0) * target.size(1))

verbose = True

for epoch in range(nb_epochs):
    print('Epoch:', epoch)

    losses = []
    clock = time.time()

    for i in range(nb_graphs):
        if verbose: print('--- Processing new graph! ---')

        # We do optimizer call only after completely processing the graph
        graph, states = graphs[i], history_dataset[i]
        # extract adjacency matrix
        edges_mat = nx.to_numpy_matrix(graph)
        if verbose: print('edges_mat shape:', edges_mat.shape)
        # Convert graph to DGL
        graph = dgl.DGLGraph(graph)

        # set states to fixed lenght with termination boolean to be able to compute termination error
        termination = np.zeros(states.shape[0])
        termination[-1] = 1
        assert max_steps >= states.shape[0]
        if states.shape[0]<max_steps:
            pad_idx = [(0,0) for i in range(states.ndim)]
            pad_idx[0] = (0, max_steps-states.shape[0])
            states = np.pad(states, pad_idx, 'edge')
            termination = np.pad(termination, (0, max_steps-termination.size), 'constant', constant_values=(1))

        if verbose: print('states shape (after reshape):', states.shape)
        if verbose: print('termination shape (after reshape):', termination.shape)

        states = torch.from_numpy(states)
        edges_mat = torch.from_numpy(edges_mat)

        # What exatly do we want as input?
        preds, pred_stops = model(graph, states, edges_mat)

        loss = nll_gaussian(preds, states) + ((pred_stops-termination)**2).sum()/max_steps # MSE of output and states + termination loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('Epoch run in:', time.time()-start)
    clock = time.time()
    print(mean(losses))
