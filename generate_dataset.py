import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from dfs import DFS
from graph_generation import GraphGenerator


class DatasetGenerator:

    def __init__(self):
        self.graph_generator = GraphGenerator()

    def run(self, graph_type, nb_graphs, nb_nodes, algorithm_type):
        graphs = []
        dataset = []
        next_nodes = []

        for _ in range(nb_graphs):
            
            if algorithm_type == 'DFS':
                dfs = DFS()
                graph = self.graph_generator.gen_graph_type(nb_nodes, graph_type)
                graphs.append(graph)
                history, _ = dfs.run(graph)
                dataset.append(history)
                # Generate the "next node" data
                next_nodes.append(np.asarray([np.where(history[i]-history[i+1]>0, 1, 0) for i in range(history.shape[0]-1)]))

        return graphs, np.asarray(dataset), np.asarray(next_nodes)


if __name__ == '__main__':
    graph_type = 'erdos_renyi'
    nb_graphs = 3
    nb_nodes = 8
    algorithm_type = 'DFS'

    data_gen = DatasetGenerator()
    graphs, dataset, next_nodes = data_gen.run(graph_type, nb_graphs, nb_nodes,
                                  algorithm_type)

    print(dataset, [np.argmax(next_node, axis=1) for next_node in next_nodes])

