#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FileLoader import FileLoader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
fl = FileLoader()
X_train, y_train, X_test, y_test = fl.load_x_and_y()

features = ["num_coocs", "arr_leav", "diversity", "unique_coocs", "weighted_frequency", "coocs_w", "countries_in_common", "num_common_travels"]

f = plt.figure()


meets = X_train[np.where(y_train == 0)]
nonmeets = X_train[np.where(y_train == 1)]

for index, feature in enumerate(features):
	ax = f.add_subplot(4,2,index+1)
	sns.distplot(meets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="blue", kde=False, hist_kws={"alpha":0.5})
	sns.distplot(nonmeets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="red", kde=False, hist_kws={"alpha":0.5})

f.tight_layout()
plt.show()