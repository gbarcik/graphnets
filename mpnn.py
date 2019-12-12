# Define the MNN model that will be trained

# Part of the code is based on https://github.com/timlacroix/nri_practical_session

import dgl
from generate_dataset import DatasetGenerator
import torch
import torch.nn as nn

# Do not use DGL in the end?

# Define the linear projection module
class Linear_layer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Linear_layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # perform linear transformation
        return self.linear(inputs)


# A first NN for a single algorithm
# Define the MPNN module
# (Use of DFS for the first exemple)
class MPNN(nn.Module):
    # Expects dgl graphs as inputs
    def __init__(self, in_feats, hidden_feats, edge_feats, out_feats):
        super(MPNN, self).__init__()
        self.encoder = Linear_layer(in_feats + hideen_feats, hidden_feats)
        self.M = Linear_layer( hidden_feats * 2 + edge_feats, hidden_feats)
        self.U = Linear_layer(hidden_feats * 2 , hidden_feats)
        self.decoder = Linear_layer(hidden_feats * 2 , in_feats)
        self.termination = Linear_layer(hidden_state * 2 , 2) # Find a way to have only 2 outputs whatever the graph size is

    def gcn_message(edges, edges_mat):
        # The argument is a batch of edges.
        # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
        z_src = edges.src['z']
        z_dst = edges.dst['z']

        msg = self.M(torch.cat([z_src, z_dst, edges_mat], 1))
        return {'msg' : msg}

    def gcn_reduce(nodes):
        # The argument is a batch of nodes.
        # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
        return {'u_input' : torch.max(nodes.mailbox['msg'], dim=1)}


    # A step corresponds to 1 iteration of the network:
    # Giving the state of the graph after one iteration of the algrithm
    def step(self, graph, inputs, edges_mat, hidden):

        n_atoms = edges_mat.size(0)
        id1 = torch.LongTensor(sum([[i] * n_atoms for i in range(n_atoms)], []))
        id2 = torch.LongTensor(sum([list(range(n_atoms)) for i in range(n_atoms)], []))

        # Encoding x^t and h^{t-1}
        z = self.encoder(torch.cat([inputs, hidden], 1))
        graph.ndata['z'] = z

        # stack z and edges
        stack = torch.cat([
                torch.index_select(z, 0, id1),
                torch.index_select(z, 0, id2),
                edges,
            ], 1)

        # Processor

        messages = self.M(stack)
        # Extract the aggregation(max) of messages at each position
        # Easier with DGL:
        g.send(g.edges(), lambda x: gcn_message(x, edges_mat))
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        new_hidden = self.U(torch.cat([z, g.ndata['u_input']], 1))

        # Stoping criterion for the next step
        stop = nn.sigmoid(self.termination(torch.cat([ne_hidden, torch.mean(new_hidden, keepdim=True)], 1)))

        # Decoder
        new_state = self.decoder(torch.cat([new_hidden, z], 1))

        return new_state, new_hidden, stop
        

    # Iterate steps until completion
    def forward(self, states, edges_mat):

        hidden = torch.zeros(edges_mat.size(0), edges_mat.size(1), self.n_hid).cuda()

        pred_all = [states[0]]
        pred_stop = [(0,0)]

        for i in range(len(states)):
            new_state, hidden, stop = self.step(pred_all[i], edges_mat, hidden)

            pred_all.append(new_state)
            pred_stop.append(stop)
        
        preds = torch.stack(pred_all, dim=1)
        preds_stop = torch.stack(pred_stop, dim=1)

        return preds, preds_stop