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
fig = plt.figure()
ax = fig.add_subplot(111)
sns.set_style("whitegrid")
plt.title("CDF of degree", color='white')
plt.plot(sorted_data, yvals)
plt.xlabel("Degree")
plt.ylabel("Probability")
# colors
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

fig.savefig("japan.png", transparent=True)
plt.show()