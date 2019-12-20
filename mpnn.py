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

    def forward(self, inputs):
        # perform linear transformation
        return self.linear(inputs)


# A first NN for a single algorithm
# Define the MPNN module
# (Use of DFS for the first exemple)
# !! Implemented with sum of received messages TODO: set max instead
class MPNN(nn.Module):
    # Expects dgl graphs as inputs
    def __init__(self, in_feats, hidden_feats, edge_feats, out_feats, useCuda=False):
        super(MPNN, self).__init__()
        self.n_hid = hidden_feats
        self.encoder = Linear_layer(in_feats + hidden_feats +1, hidden_feats) # +1 is for the weights (needed so far, might be removed later)
        self.M = Linear_layer( hidden_feats * 2 + edge_feats, 32)
        self.U = Linear_layer(hidden_feats * 2 , hidden_feats)
        self.decoder = Linear_layer(hidden_feats * 2 , in_feats)
        self.termination = Linear_layer(hidden_feats , 1) # Find a way to have only 2 outputs whatever the graph size is
        self.useCuda = useCuda

    def compute_send_messages(self, edges):
        # The argument is a batch of edges.
        # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
        z_src = edges.src['z']
        #print('z_src shape:', z_src.size())
        z_dst = edges.dst['z']
        #print('z_src shape:', z_dst.size())
        #print('edges features shape:', edges.data['features'].view(-1,1).size())

        msg = self.M(torch.cat([z_src, z_dst, edges.data['features'].view(-1,1)], 1))
        return {'msg' : msg}

    def max_reduce_messages(self, nodes):
        # The argument is a batch of nodes.
        # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
        #return {'u_input' : torch.sum(nodes.mailbox['msg'], dim=1)} # for sum and mean: add ''
        return {'u_input' : torch.max(nodes.mailbox['msg'], dim=1).values}


    # A step corresponds to 1 iteration of the network:
    # Giving the state of the graph after one iteration of the algrithm
    def step(self, graph, inputs, hidden):
        
        # Helpers to stack conviniently z and e
        n_atoms = inputs.size(0)
        id1 = torch.LongTensor(sum([[i] * n_atoms for i in range(n_atoms)], []))
        id2 = torch.LongTensor(sum([list(range(n_atoms)) for i in range(n_atoms)], []))

        # Encoding x^t and h^{t-1}
        inputs = inputs.view(-1,1)
        #print('Input size in step:', inputs.size())
        #print('hidden size in step:', hidden.size())
        #print('priority size in step:', graph.ndata['priority'].size())
        inp = torch.cat([inputs, hidden, graph.ndata['priority'].view(-1,1)], 1)
        #print('concat size in step:', inp.size())
        z = self.encoder(inp)
        graph.ndata['z'] = z

        # stack z and edges
        '''#print('first z size:', torch.index_select(z, 0, id1).size())
        #print('second z size:', torch.index_select(z, 0, id2).size())
        flat_edges = edges_mat.view(edges_mat.size(0)**2, 1)
        #print('flat_edges size:', flat_edges.size())
        stack = torch.cat([
                torch.index_select(z, 0, id1),
                torch.index_select(z, 0, id2),
                flat_edges], 1)
        #print('size of stack:', stack.size())'''

        # Processor
        # without dgl: messages = self.M(stack) but hard to pass on messages
        # Extract the aggregation(max) of messages at each position
        # Easier with DGL:
        graph.send(graph.edges(), self.compute_send_messages)
        # trigger aggregation at all nodes
        graph.recv(graph.nodes(), self.max_reduce_messages)
        #print('size of z:', z.size())
        u_input = graph.ndata.pop('u_input')
        #print('size of u_input:', u_input.size())
        new_hidden = self.U(torch.cat([z, u_input], 1))

        # Stoping criterion for the next step
        H_mean = torch.mean(new_hidden, dim=0, keepdim=True)
        '''#print('H_mean size:', H_mean.size())
        H_mean_shaped = H_mean.expand(new_hidden.size(0), -1)'''
        # For now only use H_mean as the expected broadcast is unclear
        # TODO: find right way to broadcast
        loc_inp= H_mean#torch.cat([new_hidden, H_mean_shaped], 1)
        #print('loc_inp size:', loc_inp.size())
        loc_out = self.termination(loc_inp)
        #print('loc_out size:', loc_out.size())
        m = nn.Sigmoid()
        stop = m(loc_out)
        #print('stop size:', stop.size())

        # Decoder
        new_state = self.decoder(torch.cat([new_hidden, z], 1))
        #print('new_state size:', new_state.size())

        return new_state, new_hidden, stop
        

    # Iterate steps until completion
    def forward(self, graph, states, edges_mat):
        
        # Initialize hidden state at zero
        hidden = torch.zeros(states.size(1), self.n_hid).float()
        #print('Shape of hidden state:', hidden.size())

        # Store states and termination prediction
        pred_all = [states[0].view(-1,1).float()]
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

        #print('Shape of input state:', pred_all[0].size())

        # Iterate the algorithm for all steps
        for i in range(states.size(0)-1):
            new_state, hidden, stop = self.step(graph, pred_all[i], hidden)

            pred_all.append(new_state)
            pred_stop.append(stop)
            #print(pred_stop)
        
        #print(pred_all[0].size())
        preds = torch.stack(pred_all, dim=1).view(states.size(0)-1,states.size(0))
        #print(preds.size())
        preds_stop = torch.stack(pred_stop, dim=1)

        return preds, preds_stop