import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

class GraphGenerator:
    def __init__(self):
        pass

    def gen_graph_type(self, nb_nodes, graph_type, set_weights=False):
        g = None

        if graph_type == 'gn_graph':
            g = nx.gn_graph(nb_nodes)

        if graph_type == 'ladder':
            g = nx.ladder_graph(nb_nodes)

        if graph_type == 'grid':
            g = nx.grid_2d_graph(nb_nodes, nb_nodes // 2 + 1)

        if graph_type == 'erdos_renyi':
            p = min(np.log(nb_nodes)/nb_nodes, 0.5)
            g = nx.erdos_renyi_graph(n=nb_nodes, p=p, directed=True)

        if graph_type == 'barabasi_albert':
            nb_neighs = 5
            g = nx.barabasi_albert_graph(n=nb_nodes, m=nb_neighs)

        if graph_type == '4_caveman':
            # l (int) – Number of groups
            # k (int) – Size of cliques
            # p (float) – Probabilty of rewiring each edge.
            g = nx.relaxed_caveman_graph(l=4, k=5, p=0.3)

        if g is None:
            raise ValueError
        else:
            # give a priority to each node
            priorities = {i: p for i, p in
                          enumerate(np.random.uniform(0.2, 1, len(g.nodes)))}
            nx.set_node_attributes(g, priorities, name='priority')

            # give a weight to each edge
            if set_weights:
                weights = np.random.uniform(0.2, 1, len(g.edges))
                weights = {e: weights[i] for i, e in enumerate(g.edges)}
                nx.set_edge_attributes(g, weights, 'weight')
            return g


if __name__ == '__main__':
    gen = GraphGenerator()
    g = gen.gen_graph_type(10, 'ladder')
    nx.draw(g.to_directed())
    plt.show()
    print(nx.adjacency_matrix(g).todense())
    print(g.nodes(data=True))

