# Implements the "clasical" computation of DFS

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graph_generation import GraphGenerator


class DFS:

    def __init__(self):
        pass

    def decode_last_state(self, x):
        nb_seen = np.sum(np.where(x < 0, 1, 0))
        sort = np.argsort(np.where(x < 0, -x, float('inf')))
        sort[nb_seen:] = -1
        return sort

    def run(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as root for the DFS

        Returns:
        --------
        The history of x (states) when executing the DFS algorithm, and the DFS
        output
        '''

        E = nx.to_numpy_matrix(graph)
        x = self.initialize_x(graph, root)
        history = [x.copy()]
        import pdb; pdb.set_trace()
        while np.max(x) > 0:
            x = self.iter_DFS(graph, x, E)
            history.append(x.copy())

        return np.asarray(history), self.decode_last_state(x)


    def initialize_x(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as a root for the DFS

        Returns:
        --------
        Initialized numpy representation of the graph, as used by our DFS implementation
        '''

        nb_nodes = graph.number_of_nodes()
        x = np.zeros((nb_nodes))
        x[root] = 1

        return x


    def iter_DFS(self, graph, x, E):
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
        Modifies x, using our DFS algorithm
        '''

        i0 = np.argmax(x) # Select the node with highest rank

        next_label = x[i0] + 1 # Detect rank to assign

        x[i0] = np.amin(x) - 1 # Mark node as seen. Implicitely encodes the position in which the node was seen

        neigh = np.argwhere(E[i0] > 0)[:,1] # Select the neighbours of this node

        neigh = sorted(neigh, key=lambda id: graph.nodes[id]['priority'])

        for ind in neigh:
            # If son was not explored, update it it
            if x[ind] >= 0: # Not == 0: we want to update the node's sons priority even if they are son of a shallower node
                x[ind] = next_label # Mark the sons with highest rank, so that it is explored in priority
                next_label += 1 # Update highest rank

        return x


if __name__=="__main__":
    # graph = nx.balanced_tree(2,3)
    root= 2
    generator = GraphGenerator()
    graph = generator.gen_graph_type(10, 'erdos_renyi')

    dfs = DFS()

    hs, output = dfs.run(graph)
    print('dfs output: {}'.format(output))
    print(hs)

    labels = dict((n, [n, np.around(d['priority'], decimals=2)]) for n, d in graph.nodes(data=True))
    nx.draw(graph, labels=labels)
    plt.show()

