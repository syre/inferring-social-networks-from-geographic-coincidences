#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

directory = os.path.join("..", "data", "sweden.graphml")

sweden_graph = nx.read_graphml(directory)
degrees = list(sweden_graph.degree().values())
sorted_data = np.sort(degrees)
yvals = np.arange(len(sorted_data))/float(len(sorted_data))
sns.set_style("whitegrid")
plt.title("CDF of degree")
plt.plot(sorted_data, yvals)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.show()