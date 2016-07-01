#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import networkx as nx
import seaborn
import matplotlib.pyplot as plt

directory = os.path.join("..", "data", "sweden.graphml")

sweden_graph = nx.read_graphml(directory)
degrees = list(sweden_graph.degree().values())
print(degrees)
seaborn.distplot(degrees, kde=False)
plt.xlim(0, 250)
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()