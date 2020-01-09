# Performs a training of the current MNN model

import dgl
from generate_dataset_2 import DatasetGenerator
import torch
import torch.nn as nn
from mpnn_2 import MPNN
import time
import networkx as nx
import numpy as np

# Setting the seed for replicability
import random
random.seed(32)

use_cuda = torch.cuda.is_available()

#####################################################
# --- Training parameters
#####################################################

# For the training hyperparameters, insire from paper
nb_epochs = 50
nb_features = 32
lr = 0.0005

# Datasets parameters
algorithm_type = 'DFS'

nb_graphs = {}
nb_nodes = {}
graph_types = {}

# Allows to generate graphs of different sizes in each dataset
nb_graphs['train'] = [40]
nb_nodes['train'] = [20]
graph_types['train'] = ['erdos_renyi']

nb_graphs['test'] = [20]
nb_nodes['test'] = [20]
graph_types['test'] = ['erdos_renyi']

# maximum number of steps before stopping
max_steps = max(max(nb_nodes['train']), max(nb_nodes['test'])) + 1

####################################################
# --- Data generation
####################################################

start = time.time()
data_gen = DatasetGenerator()

data = {
    'train': [],
    'test': []
}

# Prepare the data in an easily exploitable format
for phase in ['train', 'test']:
    # Generate the right number of graphs of each size
    for idx, nb_g in enumerate(nb_graphs[phase]):

        nb_n = nb_nodes[phase][idx]
        graph_type = graph_types[phase][idx]

        graphs, next_nodes = data_gen.run(graph_type, nb_g, nb_n, algorithm_type)

        terminations = []
        edges_mats = []
        next_nodes = list(next_nodes)

        # Do all the necessary transforms on the data
        for i in range(nb_g):
            states = np.asarray(next_nodes[i])
            termination = np.zeros(len(states))
            termination[-1] = 1

            # Adding self edge to every node, as in the paper
            for j in range(nb_n):
                graphs[i].add_edge(j, j)

            assert max_steps >= len(states)

            terminations.append(termination)
            edges_mats.append(nx.to_numpy_matrix(graphs[i]))
            g = dgl.DGLGraph()
            g.from_networkx(graphs[i], node_attrs=['priority'])
            graphs[i] = g

        data[phase] += [(graphs[i], edges_mats[i], next_nodes[i], terminations[i]) for i in range(nb_g)]

train_data = data['train']
test_data = data['test']

print('Dataset created in:', time.time()-start)
clock = time.time()

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


def next_state_accuracy(preds, targets):
    # Evaluates the average accuracy in predicting the next node
    next_node_pred = torch.argmax(preds, axis=-1)
    nb_false = torch.nonzero(targets-next_node_pred).size(0)
    return (targets.shape[0]-nb_false) / targets.shape[0]

verbose = False

teacher_forcing = True

if teacher_forcing:
    print('Using teacher forcing!')

for epoch in range(nb_epochs):
    print('Epoch:', epoch)

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []
    test_exact_terminations = 0

    clock = time.time()

    model.train()

    for batch_idx, (graph, edges_mat, states, termination) in enumerate(train_data):

        states = torch.from_numpy(np.asarray(states))
        edges_mat = torch.from_numpy(edges_mat)
        termination = torch.from_numpy(termination)

        if states.shape[0] > 1:
            # if more than 1 state, prepare the target of the network
            target = []
            target.extend([np.where(states[i]-states[i-1])[0] for i in range(1, states.shape[0])])
            target = np.hstack(target)
            target = torch.LongTensor(target)
            if use_cuda: target = target.cuda()

        if use_cuda:
            states, edges_mat, termination = states.cuda(), edges_mat.cuda(), termination.cuda()

        if verbose:
                print('--- Processing new graph! ---')
                print('edges_mat shape:', edges_mat.shape)
                print('states shape (after reshape):', states.shape)
                print('termination shape (after reshape):', termination.shape)

        # We do optimizer call only after completely processing the graph
        preds, pred_stops = model(graph, states, edges_mat)

        # Compare the components of the loss for tuning
        if states.shape[0] > 1:
            loss = nn.CrossEntropyLoss()
            output = loss(preds, target)
        else:
            # Sometimes the algorithm is already terminated when starting, in which case there is nothing to compare
            output = torch.tensor([0]).type(torch.FloatTensor)
            if use_cuda: output = output.cuda()

        loss2 = nn.BCELoss()
        output += loss2(pred_stops.view(-1, 1), termination.float().view(-1, 1))

        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        train_losses.append(output.item())
        if states.shape[0] > 1: train_accuracies.append(next_state_accuracy(preds, target))

    print('Train epoch run in:', time.time()-clock)
    clock = time.time()
    print(' Training Loss:', np.mean(np.asarray(train_losses)))
    print('Train average accuracy:', np.mean(np.asarray(train_accuracies)))

    model.eval()

    for batch_idx, (graph, edges_mat, states, termination) in enumerate(test_data):

        states = torch.from_numpy(np.asarray(states))
        edges_mat = torch.from_numpy(edges_mat)
        termination = torch.from_numpy(termination)

        target = None
        if states.shape[0] > 1:
            # if more than 1 state, prepare the target of the network
            target = []
            target.extend([np.where(states[i]-states[i-1])[0] for i in range(1, states.shape[0])])
            target = np.hstack(target)
            target = torch.LongTensor(target)
            if use_cuda: target = target.cuda()

        if use_cuda:
            states, edges_mat, termination = states.cuda(), edges_mat.cuda(), termination.cuda()

        if verbose:
                print('--- Processing new graph! ---')
                print('edges_mat shape:', edges_mat.shape)
                print('states shape (after reshape):', states.shape)
                print('termination shape (after reshape):', termination.shape)


        # We do optimizer call only after completely processing the graph
        preds, pred_stops = model.predict(graph, states, edges_mat)

        # Compare the components of the loss for tuning

        # In tests, need to compare the right lenghts
        if preds is not None and states.shape[0] > 1:
            comparable_lenght = min(preds.size()[0], target.size()[0])
            test_exact_terminations += (preds.size()[0] == target.size()[0])
        else:
            comparable_lenght = 0
            test_exact_terminations += (states.shape[0] == 1)

        if states.shape[0] > 1 and preds is not None: # 
            loss = nn.CrossEntropyLoss()
            output = loss(preds[:comparable_lenght], target[:comparable_lenght])
        else:
            # Sometimes the algorithm is already terminated when starting, in which case there is nothing to compare
            output = torch.tensor([0]).type(torch.FloatTensor)
            if use_cuda: output = output.cuda()

        loss2 = nn.BCELoss()
        output += loss2(pred_stops.view(-1, 1)[:comparable_lenght+1], termination.float().view(-1, 1)[:comparable_lenght+1])

        test_losses.append(output.item())
        if states.shape[0] > 1 and preds is not None: test_accuracies.append(next_state_accuracy(preds[:comparable_lenght], target[:comparable_lenght]))


    if epoch<nb_epochs-1:
        print('Test ex')
        print('states:', states)
        print('pred:', preds)

    print('Test epoch run in:', time.time()-clock)
    clock = time.time()
    print(' Test Loss:', np.mean(np.asarray(test_losses)))
    print('Test average accuracy:', np.mean(np.asarray(test_accuracies)))
    print('Test exact termination accuracy:', test_exact_terminations/len(test_data))


print('states:', states)
print('pred:', preds)
print('termination:', termination)
print('pred_stops:', pred_stops)