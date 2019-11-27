# Implements the "clasical" computation of DFS

import os
import numpy as np
import networkx as nx
import time
from IPython.display import clear_output

def prepare_x_DFS(graph, root=0):
    '''
    Parameters
    ----------
    graph: NetworkX Graph instance
        The graph on which the algorithm should be run
    source: index of the node that should be used as a root for the DFS
    
    Returns:
    --------
    Initialized numpy representation of the graph, as used by our DFS implementation
    '''

    nb_nodes = graph.number_of_nodes()
    x = np.zeros((nb_nodes))
    x[root] = 1

    return x


def iter_DFS(x, E):
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
    next_label = x[i0]+1 # Detect rank to assign

    x[i0] = np.amin(x)-1 # Mark node as seen. Implicitely encodes the position in which the node was seen
    
    neigh = np.argwhere(E[i0]==1)[:,1] # Select the neighbourgs of this node
        
    for ind in neigh:
        # If son was not explored, update it it
        if x[ind] == 0:
            x[ind] = next_label # Mark the sons with highest rank, so that it is explored in priority
            next_label += 1 # Update highest rank
    


if __name__=="__main__":
    graph = graph_ex = nx.balanced_tree(2,3)
    root = 2

    E = nx.to_numpy_matrix(graph)
    x = prepare_x_DFS(graph, root)
    print(x)

    while np.amax(x)>=0:
        iter_DFS(x,E)
        print(x)
        time.sleep(5)
        clear_output()