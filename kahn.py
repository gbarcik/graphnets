# Implements the "clasical" computation of Kahn

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graph_generation import GraphGenerator


class Kahn:

    def __init__(self):
        pass

    def decode_last_state(self, x):
        return x[:,1]

    def run(self, graph):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
            The graph on which the algorithm should be run

        Returns:
        --------
        The history of x (states) when executing the Kahn algorithm, and the execution list
        output
        '''

        E = nx.to_numpy_matrix(graph)
        x = self.initialize_x(graph)
        history = [x.copy()]

        while np.amin(x[:,2])<1:
            print(x)
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

        x = np.array([(get_degree(E,i), -1, -1) for i in range(nb_nodes)])

        # Labels will encode the order of execution
        label = 1
        for i in range(len(x)):
            if x[i][0] == 0:
                x[i][1] = label
                x[i][2] = 0
                label += 1

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
    
        next_label = np.amax(x[:,1])+1
        assert next_label >= 1 # If no node is free at the begining, problem cannot be solved
        if np.sum(np.where(x[:,2]==0, 1, 0))==0:
            # Algorithm is stuck
            # Interupt iterations 
            x[:,2] = 1
        node_to_free = np.argmin(np.where(x[:,2]==0, x[:,1], float('inf')))

        x[node_to_free, 2] = 1 # Set the node as seen

        neigh = np.argwhere(E[node_to_free]==1) # Get all nodes the depend on its execution

        for ind in neigh[:,1]:
            x[ind, 0] -= 1 # Decrease the number of constrain for the neighbourg
            if x[ind, 0] == 0:
                # If the degree reaches zero, set a label to the node: it is ready to be processed
                x[ind, 1] = next_label
                x[ind, 2] = 0
                next_label += 1
        
        return x




if __name__=="__main__":
    # graph = nx.balanced_tree(2,3)
    root= 2
    generator = GraphGenerator()
    graph = generator.gen_graph_type(10, 'erdos_renyi')

    kahn = Kahn()

    hs, output = kahn.run(graph)
    print(kahn.run(graph)[1])

    #import pdb; pdb.set_trace()

    E = nx.to_numpy_matrix(graph)
    x = kahn.initialize_x(graph)
    print(x)

    while np.amin(x[:,2])<1:
        kahn.iter_Kahn(graph, x,E)
        print(x)
#        time.sleep(5)
        #clear_output()

    print('Kahn output: {}'.format(x[:,1]))

    labels = dict((n, [n, np.around(d['priority'], decimals=2)]) for n, d in graph.nodes(data=True))
    nx.draw(graph, labels=labels)
    plt.show()