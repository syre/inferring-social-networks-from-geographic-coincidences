#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FileLoader import FileLoader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
fl = FileLoader()
X_train, y_train, X_test, y_test = fl.load_x_and_y()

features = ["num_coocs", "arr_leav", "diversity", "unique_coocs", "weighted_frequency", "coocs_w", "countries_in_common", "num_common_travels"]

f, ax = plt.subplots(len(features))
f.tight_layout()
meets = X_train[y_train[y_train == 0]]
nonmeets = X_train[y_train[y_train == 1]]
print(Counter(meets[:,0]).most_common(50))
print(Counter(nonmeets[:,0]).most_common(50))
print(meets.shape)
print(nonmeets.shape)
for index, feature in enumerate(features):
	sns.distplot(meets[:,index], ax=ax[index], hist=True, norm_hist=True, axlabel=feature, color="blue", kde=False, hist_kws={"alpha":0.5})
	sns.distplot(nonmeets[:,index], ax=ax[index], hist=True, norm_hist=True, axlabel=feature, color="red", kde=False, hist_kws={"alpha":0.5})

plt.show()