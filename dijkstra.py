# Implements the "clasical" computation of Dijkstra algorithm

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graph_generation import GraphGenerator


class Dijkstra:

    def __init__(self):
        pass

    def decode_last_state(self, x):
        # to do: recover solution from x
        return None

    def run(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as the source for Dijkstra

        Returns:
        --------
        The history of x (states) when executing the Dijkstra algorithm, and
        the Dijkstra output
        '''

        E = nx.to_numpy_matrix(graph)
        x = self.initialize_x(graph, root)
        history = [x.copy()]
        print(x)
        # stop when the  smallest tentative distance among unvisited nodes +inf
        while np.max(x[x[:, 0] == 1][:, 1]) < np.float('inf'):
            x = self.iter_dijkstra(graph, x, E)
            history.append(x.copy())
            print(x)

        return np.asarray(history), self.decode_last_state(x)


    def initialize_x(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as the source for Dijkstra

        Returns:
        --------
        Initialized numpy representation of the graph, as used by our Dijkstra implementation
        x[i] contains two fields (unvisited (bool), distance to source (float))
        '''

        nb_nodes = graph.number_of_nodes()
        x = np.ones((nb_nodes, 2))
        x[:, 1] = np.float('inf')
        x[root] = [1, 0]

        return x


    def iter_dijkstra(self, graph, x, E):
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
        Modifies x, using our Dijkstra algorithm
        '''
        # minimum distance of unvisited nodes
        min_dist = np.min(x[x[:, 0] == 1][:, 1])
        i0 = np.argwhre(x[:, 1] == min_dist)[0][0]

        # select the neighbours of this node
        neigh = np.argwhere(E[i0]>0)[:,1]

        for ind in neigh:
            # update only unvisited nodes
            if x[ind][0] == 1:
                x[ind][1] = min(x[ind][1], E[i0, ind] + x[i0][1])

        # mark current node as visited
        x[i0][0] = 0

        return x


if __name__=="__main__":
    root= 2
    generator = GraphGenerator()
    graph = generator.gen_graph_type(10, 'erdos_renyi')

    dijkstra = Dijkstra()

    hs, output = dijkstra.run(graph)
    print(dijkstra.run(graph)[1])

    import pdb; pdb.set_trace()

    labels = dict((n, [n, np.around(d['priority'], decimals=2)]) for n, d in graph.nodes(data=True))
    nx.draw(graph, labels=labels)
    plt.show()

