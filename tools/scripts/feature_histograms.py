#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FileLoader import FileLoader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file
fl = FileLoader()

X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")
X_train = X_train.toarray()
X_test = X_test.toarray()

features = ["num_coocs", "num_unique_coocs", "diversity", "weighted_frequency",
 "sum_weekends", "sum_evenings", "coocs_w", "mutual_cooccurrences", "specificity"]

f = plt.figure()


meets = X_train[np.where(y_train == 1)]
nonmeets = X_train[np.where(y_train == 0)]
#sns.set(font_scale=2.5)
#sns.distplot(meets[:,5], hist=True, norm_hist=True, axlabel=features[5], color="blue", kde=False, hist_kws={"alpha":0.5})
for index, feature in enumerate(features):
    ax = f.add_subplot(6,2,index+1)
    sns.distplot(meets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="blue", kde=False, hist_kws={"alpha":0.5})
    sns.distplot(nonmeets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="red", kde=False, hist_kws={"alpha":0.5})
    #[item.set_fontsize(35) for item in [ax.yaxis.label, ax.xaxis.label]]
    #ax.title.set_fontsize(40)
    #[item.set_fontsize(28) for item in ax.get_xticklabels() + ax.get_yticklabels()]

f.tight_layout()
plt.show()