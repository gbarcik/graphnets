# Define the MNN model that will be trained

# Part of the code is based on https://github.com/timlacroix/nri_practical_session

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting the seed for replicability
import random
random.seed(33)

# Do not use DGL in the end?

# A first NN for a single algorithm
# Define the MPNN module
# (Use of DFS for the first exemple)

class MPNN(nn.Module):
    # Expects dgl graphs as inputs
    def __init__(self, in_feats, hidden_feats, edge_feats, out_feats, useCuda=False):
        super(MPNN, self).__init__()
        self.n_hid = hidden_feats
        self.encoder = nn.Linear(in_feats + hidden_feats +1, hidden_feats) # +1 is for the weights (needed so far, might be removed later)
        self.M = nn.Linear( hidden_feats * 2 + edge_feats, 32)
        self.U = nn.Linear(hidden_feats * 2 , hidden_feats)
        self.decoder = nn.Linear(hidden_feats * 2 , in_feats) # "first" version, does not account for next node prediction
        self.termination = nn.Linear(hidden_feats , 1) # Find a way to have only 1 outputs whatever the graph size is
        self.useCuda = useCuda

    def compute_send_messages(self, edges):
        # The argument is a batch of edges.
        # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
        z_src = edges.src['z']
        z_dst = edges.dst['z']

        msg = self.M(torch.cat([z_src, z_dst, edges.data['features'].view(-1,1)], 1))
        return {'msg' : msg}

    def max_reduce_messages(self, nodes):
        # The argument is a batch of nodes.
        # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
        return {'u_input' : torch.max(nodes.mailbox['msg'], dim=1).values}


    # A step corresponds to 1 iteration of the network:
    # Giving the state of the graph after one iteration of the algrithm
    def step(self, graph, inputs, hidden):
        
        # Encoding x^t and h^{t-1}
        inputs = inputs.view(-1,1).float()
        inp = torch.cat([inputs, hidden, graph.ndata['priority'].view(-1,1)], 1)
        z = self.encoder(inp)
        graph.ndata['z'] = z

        # Processor
        graph.send(graph.edges(), self.compute_send_messages)
        # trigger aggregation at all nodes
        graph.recv(graph.nodes(), self.max_reduce_messages)

        u_input = graph.ndata.pop('u_input')

        new_hidden = self.U(torch.cat([z, u_input], 1))

        # Stoping criterion for the next step
        H_mean = torch.mean(new_hidden, dim=0, keepdim=True)        
        loc_out = self.termination(H_mean)
        m = nn.Sigmoid()
        stop = m(loc_out)
        stop = torch.max(stop).view((1,1))

        # Decoder
        new_state = self.decoder(torch.cat([new_hidden, z], 1))
        new_state = new_state.masked_fill(inputs.bool(), float('-inf'))
        softmax = F.softmax(new_state, 0)        

        return softmax, new_hidden, stop
        

    # Iterate steps until completion
    def forward(self, graph, states, edges_mat):
        # Initialize hidden state at zero
        hidden = torch.zeros(states.size(1), self.n_hid).float()

        # Store states and termination prediction
        pred_states = []
        pred_stop = [torch.tensor([[0]]).float()]

        # set all edges features inside graph (for easier message passing)
        edges_features = []
        for i in range(graph.edges()[0].size(0)):
            # Extract the features of each existing edge
            edges_features.append(edges_mat[graph.edges()[0][i], graph.edges()[1][i]])
        
        graph.edata['features'] = torch.FloatTensor(edges_features)

        if self.useCuda:
            graph.edata['features'] = graph.edata['features'].cuda()
            graph.ndata['priority'] = graph.ndata['priority'].cuda()
            hidden = hidden.cuda()
            pred_stop = [torch.tensor([[0]]).float().cuda()]

        # Iterate the algorithm for all steps
        for i in range(states.size(0)): # -1
            new_state, hidden, stop = self.step(graph, states[i], hidden)

            pred_states.append(new_state)
            pred_stop.append(stop)
                
        if len(pred_states) == 1: # 0
            preds = torch.empty(1)
            preds_stop = torch.stack([pred_stop[1]], dim=1)
        else:
            preds = torch.stack(pred_states[:-1], dim=0).view(-1, states.size(1))
            preds_stop = torch.stack(pred_stop[:-1], dim=1)
        
        return preds, preds_stop


    # Iterate steps until completion
    def predict(self, graph, states, edges_mat):
        # Initialize hidden state at zero
        hidden = torch.zeros(states.size(1), self.n_hid).float()

        # Store states and termination prediction
        pred_states = []
        pred_stop = [torch.tensor([[0]]).float()]

        # set all edges features inside graph (for easier message passing)
        edges_features = []
        for i in range(graph.edges()[0].size(0)):
            # Extract the features of each existing edge
            edges_features.append(edges_mat[graph.edges()[0][i], graph.edges()[1][i]])
        
        graph.edata['features'] = torch.FloatTensor(edges_features)

        if self.useCuda:
            graph.edata['features'] = graph.edata['features'].cuda()
            graph.ndata['priority'] = graph.ndata['priority'].cuda()
            hidden = hidden.cuda()
            pred_stop = [torch.tensor([[0]]).float().cuda()]

        # Iterate the algorithm until termination flag
        new_state = states[0].clone() # Clone to avoid modifying ground truth data
        stop = 0
        for _ in range(graph.number_of_nodes()) :
            if stop > 0.5:
                break

            softmax, hidden, stop = self.step(graph, new_state, hidden) # was torch.Tensor(new_state.float())

            idx = torch.argmax(softmax).item()

            # loc_res is the prediction, new_state the new state
            loc_res = torch.zeros(states.size(1), 1).float()
            if self.useCuda: loc_res = loc_res.cuda()
            new_state[idx] = 1
            loc_res[idx] =1

            pred_states.append(loc_res)
            pred_stop.append(stop)

        if len(pred_states) == 1: # 0
            preds = None
            preds_stop = torch.stack([pred_stop[1]], dim=1)
        else:
            preds = torch.stack(pred_states[:-1], dim=0).view(-1, states.size(1))
            preds_stop = torch.stack(pred_stop[:-1], dim=1)
        
        return preds, preds_stop
