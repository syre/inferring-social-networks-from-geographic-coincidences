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

#X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
#X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")
#X_train = X_train.toarray()
#X_test = X_test.toarray()

X_train, y_train, X_test, y_test = fl.load_x_and_y()
test_features = ["has_two_unique_coocs", "arr_leav", "diversity", "unique_coocs", "weighted_frequency",
                 "coocs_w", "common_travels", "num_coocs", "mutual_coocs", "within_6", "same_gender",
                 "app_jaccard", "num_evenings", "num_weekends", "specificity"]

prod_features = ["num_coocs", "num_unique_coocs", "diversity", "weighted_frequency",
 "sum_weekends", "sum_evenings", "coocs_w", "mutual_cooccurrences", "specificity"]

f = plt.figure(figsize=(10, 10))
#y_train = y_train[X_train[:,0] < 100]
#X_train = X_train[X_train[:,0] < 100]
meets = X_train[np.where(y_train == 1)]
nonmeets = X_train[np.where(y_train == 0)]
#sns.set(font_scale=2.5)
#sns.distplot(meets[:,5], hist=True, norm_hist=True, axlabel=features[5], color="blue", kde=False, hist_kws={"alpha":0.5})
for index, feature in enumerate(test_features):
    ax = f.add_subplot(8,2,index+1)
    ax.set_title(feature)
    #sns.distplot(meets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="blue", kde=False, hist_kws={"alpha":0.5})
    #sns.distplot(nonmeets[:,index], ax=ax, hist=True, norm_hist=True, axlabel=feature, color="red", kde=False, hist_kws={"alpha":0.5})
    all_events = np.hstack((meets[:,index], nonmeets[:,index]))
    types = np.hstack((["did meet" for x in range(len(meets[:,index]))], ["did not meet" for x in range(len(nonmeets[:,index]))]))
    sns.boxplot(x=types, y=all_events, ax=ax)
    
    #[item.set_fontsize(35) for item in [ax.yaxis.label, ax.xaxis.label]]
    #ax.title.set_fontsize(40)
    #[item.set_fontsize(28) for item in ax.get_xticklabels() + ax.get_yticklabels()]

f.tight_layout()
plt.show()