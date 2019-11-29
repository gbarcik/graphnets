# Implements the "clasical" computation of Kahn

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graph_generation import GraphGenerator


class Kahn:

    def __init__(self):
        pass

    def decode_last_state(self, x):
        # argsort of the last iteration, with handeling of the unseen nodes
        nb_seen = np.sum(np.where(x[:,0] < 0, 1, 0))
        sort = np.argsort(np.where(x[:,0] < 0, -x[:,0], float('inf')))
        sort[nb_seen:] = -1
        return sort

    def run(self, graph):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
            The graph on which the algorithm should be run

        Returns:
        --------
        The history of x (states) when executing the Kahn algorithm, and the
        execution list output
        '''

        E = nx.to_numpy_matrix(graph)
        x = self.initialize_x(graph)
        history = [x.copy()]

        # Stopping condition is when no node can be processed
        while np.any(np.isin(0, x[:,1])):
            x = self.iter_Kahn(graph, x, E)
            history.append(x.copy())

        return np.asarray(history), self.decode_last_state(x)


    def initialize_x(self, graph):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
            The graph on which the algorithm should be run

        Returns:
        --------
        Initialized numpy representation of the graph, as used by our Kahn implementation
        '''
        E = nx.to_numpy_matrix(graph)
        nb_nodes = graph.number_of_nodes()

        def get_degree(E, ind):
            return np.sum(E[:,ind])

        x = np.array([(get_degree(E,i), -1) for i in range(nb_nodes)])

        # Nodes with degree 0 are the ones that can be executed right away        
        free_idx = np.argwhere(x[:,0]==0)
        free_nodes = [free_idx[i][0] for i in range(free_idx.size)]

        if not free_nodes:
            print('No free nodes')
            return x

        for i in free_nodes:
            x[i][1] = 0

        return x


    def iter_Kahn(self, graph, x, E):
        '''
        Parameters
        ----------
        x: numpy array
            array of the node's features.
            At initialization, x[i] should be 1 for the source node and 0 otherwise
        E: numpy array
            adjacency matrix. E[i,j]=1 indicates a edge from node i to node j

        Returns
        -------
        Modifies x, using our Kahn algorithm
        '''

        available_nodes = np.argwhere(x[:,1]==0)
    
        # Getting the prioritary node
        i0 = sorted([available_nodes[i][0] for i in range(available_nodes.size)], key=lambda id: graph.nodes[id]['priority'])[-1]

        m = np.amin(x[:,0])

        # Set the node as seen
        x[i0, 1] = 1
        # Store its execution time in the labels
        x[i0, 0] = m-1

        # Get all nodes the depend on its execution
        neigh = np.argwhere(E[i0] == 1)

        for ind in neigh[:,1]:
            # Decrease the degree
            x[ind, 0] -= 1

            if x[ind, 0] == 0:
                # If the degree reaches zero, set the node as able to be processed
                x[ind, 1] = 0

        return x


if __name__=="__main__":
    # graph = nx.balanced_tree(2,3)
    root= 2
    generator = GraphGenerator()

    # Ensure that we generate a directed acyclic graph
    graph = generator.gen_graph_type(10, 'gn_graph')
    print(nx.to_numpy_matrix(graph))
        

    kahn = Kahn()

    hs, output = kahn.run(graph)
    print(kahn.run(graph)[1])

    E = nx.to_numpy_matrix(graph)
    x = kahn.initialize_x(graph)
    print(x)

    while np.any(np.isin(0, x[:,1])):
        kahn.iter_Kahn(graph, x,E)
        print(x)

    print('Kahn output: {}'.format(kahn.decode_last_state(x)))

    labels = dict((n, [n, np.around(d['priority'], decimals=2)]) for n, d in graph.nodes(data=True))
    nx.draw(graph, labels=labels)
    plt.show()

