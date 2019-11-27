# Extending the Neural Graph Algorithm Executor

## Description

*Topic*: algorithms/data structures, graph neural networks, learning-to-execute

*Category*: implementation + research

In recent work [1, 2], the utility of *graph neural networks* as algorithm executors has been demonstrated. Naturally, one might wish to extend these insights to a broader space of algorithms. In this project, you would be expected to teach neural networks to execute one or more additional parallel or sequential graph algorithms, previously unexplored by related work (some suggestions: depth-first search, Dijkstra’s algorithm or topological sorting). The students will have freedom to choose the depth/breadth of the project (e.g. whether to focus just on one algorithm with in-depth studies, or to explore multiple algorithms at once).

## Setup

1. Clone the main repo
```sh
https://github.com/gabrielbarcik/graphnets.git
```

2. Create the conda environment (graphnet) `environment_nobuilds.yml`.

```sh
conda env create -f environment_nobuilds.yml
```


## Resources
[1] Veličković, P., Ying, R., Padovano, M., Hadsell, R. and Blundell, C. (2019). Neural Execution of Graph Algorithms. arXiv preprint arXiv:1910.10593

[2] Anonymous (2019). Neural Execution Engines. Submitted to ICLR 2020.

## Authors


* **Gabriel Fedrigo Barcik** - [mail](gbarcike@gmail.com)
* **Louis Dumont** - [mail](louis.dumont@eleves.enpc.fr)

