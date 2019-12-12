# Performs a training of the current MNN model
# TODO: store the generated graphs and data to make sure the experiments are replicable
# TODO: store results of experiments in a file

import dgl
from generate_dataset import DatasetGenerator
import torch
import torch.nn as nn
from mpnn import MPNN
import time

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
graphs, dataset = data_gen.run(graph_type, nb_graphs, nb_nodes,
                                algorithm_type)

print('Dataset created in:', time.time()-start)
clock = time.time()

# Inspect some shapes
for i in range(10):
    print(dataset[i].shape)

# Item is an array of size batch size
# each idem is a history of shape (nb_steps, nb_nodes)

# We now have in our possession a full dataset on which to train


###################################################

model = MPNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

for epoch in range(nb_epochs):
    print('Epoch:', epoch)

    losses = []
    clock = time.time()

    for i in range(nb_graphs):

        # We do optimizer call only after completely processing the graph
        graph, states = graphs[i], dataset[i]
        # Convert all data to DGL
        graph = dgl.DGLGraph(graph)
        for i in range(len(states)):
            states[i] = dgl.DGLGraph(states[i])

        # TODO: set states to fixed lenght with termination boolean to be able to compute termination error

        output = model(graph)

        loss = # MSE of output and states + termination

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('Epoch run in:', time.time()-start)
    clock = time.time()
    print(mean(losses))
