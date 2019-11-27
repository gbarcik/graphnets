import networkx as nx
from matplotlib import pyplot as plt

class GraphGenerator:
    def __init__(self):
        pass

    def gen_graph_type(self, nb_nodes, graph_type):
        if graph_type == 'ladder':
            return nx.ladder_graph(nb_nodes)

        if graph_type == 'grid':
            return nx.grid_2d_graph(nb_nodes, nb_nodes // 2 + 1)

        if graph_type == 'erdos_renyi':
            p = min(np.log(nb_nodes)/nb_nodes, 0.5)
            return nx.erdos_renyi_graph(n=nb_nodes, p=p, directed=True)

        if graph_type == 'barabasi_albert':
            nb_neighs = 5
            return nx.barabasi_albert_graph(n=nb_nodes, m=nb_neighs)

        if graph_type == '4_caveman':
            # l (int) – Number of groups
            # k (int) – Size of cliques
            # p (float) – Probabilty of rewiring each edge.
            return  nx.relaxed_caveman_graph(l=4, k=5, p=0.3)

        raise ValueError


if __name__ == '__main__':
    gen = GraphGenerator()
    g = gen.gen_graph_type(10, 'ladder')
    nx.draw(g.to_directed())
    plt.show()
    print(nx.adjacency_matrix(g).todense())

