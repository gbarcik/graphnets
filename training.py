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

use_cuda = torch.cuda.is_available()

#####################################################
# --- Training parameters
#####################################################

# For the training hyperparameters, insire from paper
nb_epochs = 20
nb_features = 32
lr = 0.005

# Datasets parameters
graph_type = 'erdos_renyi'
nb_graphs = 200
nb_nodes = 20
algorithm_type = 'DFS'

max_steps = nb_nodes + 1 # maximum number of steps before stopping
# I added +1 as experimentally the case happends, to investigate

####################################################
# --- Data generation
####################################################

start = time.time()
data_gen = DatasetGenerator()
graphs, history_dataset, next_nodes = data_gen.run(graph_type, nb_graphs, nb_nodes,
                                algorithm_type)

print('Dataset created in:', time.time()-start)
clock = time.time()

# Prepare the data in an easily exploitable format
# It could probably be optimised with DGL

terminations = []
edges_mats = []

# Do all the necessary transforms on the data
for i in range(nb_graphs):
    states = history_dataset[i]
    nextnode_data = next_nodes[i]
    termination = np.zeros(states.shape[0])
    termination[-1] = 1
    
    # TODO?: Adding self edge to every node, as in the paper
    for j in range(nb_nodes):
        graphs[i].add_edge(j, j)

    assert max_steps >= states.shape[0]

    # set states to fixed lenght with termination boolean to be able to compute termination error
    if states.shape[0] < max_steps:
        pad_idx = [(0,0) for i in range(states.ndim)]
        pad_idx[0] = (0, max_steps-states.shape[0])
        states = np.pad(states, pad_idx, 'edge')
        nextnode_data = np.pad(nextnode_data, pad_idx, 'edge')
        termination = np.pad(termination, (0, max_steps-termination.size), 'constant', constant_values=(1))
    history_dataset[i] = states
    # For nn.CrossEntroppyLoss, next node is expected to be an index and not 1 hot
    nextnode_data = np.argmax(nextnode_data, axis=1)
    next_nodes[i] = nextnode_data
    terminations.append(termination)
    edges_mats.append(nx.to_numpy_matrix(graphs[i]))
    g = dgl.DGLGraph()
    g.from_networkx(graphs[i], node_attrs=['priority'])
    graphs[i] = g

# Take 10% of the graphs as validation
nb_val = int(0.1*nb_graphs)

train_data = [(graphs[i], edges_mats[i], history_dataset[i], terminations[i], next_nodes[i]) for i in range(nb_graphs-nb_val)]
test_data = [(graphs[i], edges_mats[i], history_dataset[i], terminations[i], next_nodes[i]) for i in range(nb_graphs-nb_val, nb_graphs)]

# Data loaders do not seem to be compatible with DGL
#train_loader = torch.utils.data.DataLoader(train_data)
#train_loader = torch.utils.data.DataLoader(test_data)

# dataset is an array of size batch size
# each idem is a history of shape (nb_steps, nb_nodes)
# We now have in our possession a full dataset on which to train


###################################################

model = MPNN(1, 32, 1, 1, useCuda=use_cuda)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

model.train()

def nll_gaussian(preds, target):
    neg_log_p = ((preds - target) ** 2)
    return neg_log_p.sum() / (target.size(0) * target.size(1))

verbose = False

teacher_forcing = True

if teacher_forcing:
    print('Using teacher forcing!')

for epoch in range(nb_epochs):
    print('Epoch:', epoch)

    losses = []
    clock = time.time()

    for batch_idx, (graph, edges_mat, states, termination, nextnodes_mat) in enumerate(train_data):

        states = torch.from_numpy(states)
        edges_mat = torch.from_numpy(edges_mat)
        termination = torch.from_numpy(termination)
        nextnodes_mat = torch.from_numpy(nextnodes_mat)
        
        if use_cuda:
                states, edges_mat, termination, nextnodes_mat = states.cuda(), edges_mat.cuda(), termination.cuda(), nextnodes_mat.cuda()

        if verbose:
                print('--- Processing new graph! ---')
                print('edges_mat shape:', edges_mat.shape)
                print('states shape (after reshape):', states.shape)
                print('termination shape (after reshape):', termination.shape)

        if teacher_forcing:

            for step in range(states.size()[1]-1):
                loc_states = states[step:step+2]
                loc_nextnode = nextnodes_mat[step]
                loc_termination = termination[step:step+2]
                preds, pred_stops, pred_nextnodes = model(graph, loc_states, edges_mat)
                pred_nextnodes = pred_nextnodes.view(-1, pred_nextnodes.size()[0])
                #print('size of nextnode pred in train', pred_nextnodes.size())

                loss = nll_gaussian(preds, (loc_states))
                loss += 100 * nn.CrossEntropyLoss()(pred_nextnodes, loc_nextnode.unsqueeze(0))
                loss += ((pred_stops-loc_termination)**2).sum()/max_steps

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                

        else:


            # We do optimizer call only after completely processing the graph
            #graph, states = graphs[i], history_dataset[i]
            # extract adjacency matrix
            #edges_mat = nx.to_numpy_matrix(graph)
            # Convert graph to DGL
            #graph = dgl.DGLGraph(graph)
            
            preds, pred_stops, pred_nextnodes = model(graph, states, edges_mat)

            #print('true', nextnodes_mat.size())
            pred_nextnodes = pred_nextnodes.view(-1, pred_nextnodes.size()[0])
            #print('pred', pred_nextnodes.size())

            # Compare the components of the loss for tuning
            loss = nll_gaussian(preds, torch.t(states))
            print('prediction loss:', nll_gaussian(preds, torch.t(states)))
            loss += 100 * nn.CrossEntropyLoss()(pred_nextnodes, nextnodes_mat)
            print('Next node prediction loss:', nn.CrossEntropyLoss()(pred_nextnodes, nextnodes_mat))
            print('pred_nextnodes', pred_nextnodes)
            print('nextnodes_mat', nextnodes_mat)
            loss += ((pred_stops-termination)**2).sum()/max_steps # MSE of output and states + termination loss
            print('termination loss:', ((pred_stops-termination)**2).sum()/max_steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    print('Epoch run in:', time.time()-clock)
    clock = time.time()
    print('Loss:', np.mean(np.asarray(losses)))

print('states:', states)
print('pred:', torch.t(preds))
print('termination:', termination)
print('pred_stops:', pred_stops)