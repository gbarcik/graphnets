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
        The history of x, p (states) when executing the Dijkstra algorithm, and
        the Dijkstra output
        '''

        E = nx.to_numpy_matrix(graph)

        # set infinity as sum(weights) + 1
        inf = sum([w[2] for w in graph.edges.data('weight')]) + 1
        print('inf set to be: {}'.format(inf))

        x, p = self.initialize_states(graph, inf, root)

        history = [x.copy(), p.copy()]

        # stop when the  smallest tentative distance among unvisited nodes +inf
        while np.min(x[x[:, 0] == 1][:, 1], initial=inf) != inf:
            x, p = self.iter_dijkstra(graph, x, p, E)
            history.append((x.copy(), p.copy()))
            print(x, p)

        return np.asarray(history)


    def initialize_states(self, graph, inf, root=0):
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
        p[i] contain an integer that represents the previous node to get to i
        '''

        nb_nodes = graph.number_of_nodes()
        x = np.ones((nb_nodes, 2))
        x[:, 1] = inf
        x[root] = [1, 0]

        p = -1 * np.ones(nb_nodes)
        p[root] = root

        return x, p


    def iter_dijkstra(self, graph, x, p, E):
        '''
        Parameters
        ----------
        x: numpy array
        array of the node's features.
        At initialization, x[i] should be 1 for the source node and 0 otherwise
        E: numpy array
        adjacency matrix. E[i,j]>0 indicates a edge from node i to node j

        Returns
        -------
        Modifies x and p using our Dijkstra algorithm
        '''

        # minimum distance of unvisited nodes
        min_dist = np.min(x[x[:, 0] == 1][:, 1])
        i0 = np.argwhere(x[:, 1] == min_dist)[0][0]

        # select the neighbours of this node
        neigh = np.argwhere(E[i0]>0)[:,1]

        for v in neigh:
            # update only unvisited nodes
            if x[v][0] == 1:
                if E[i0, v] + x[i0][1] < x[v][1]:
                    # x[v][1] = min(x[v][1], E[i0, v] + x[i0][1])
                    x[v][1] = E[i0, v] + x[i0][1]
                    p[v] = i0

        # mark current node as visited
        x[i0][0] = 0

        return x, p


if __name__=="__main__":
    root= 2
    generator = GraphGenerator()
    graph = generator.gen_graph_type(5, 'erdos_renyi', set_weights=True)

    dijkstra = Dijkstra()

    hs = dijkstra.run(graph)
    print(hs)

    labels = dict((n, [n, np.around(d['priority'], decimals=2)]) for n, d in graph.nodes(data=True))
    nx.draw(graph, labels=labels)
    pos = nx.spring_layout(graph)
    edges = nx.get_edge_attributes(graph, 'weight')
    edges = {e: np.around(w, decimals=2) for e, w in edges.items()}
    print(edges)

    plt.show()

