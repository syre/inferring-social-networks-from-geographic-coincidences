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


def hist(is_prod=True):
    if is_prod:
        X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
        X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        features = ["Number of co-occurrences (7)", "Unique co-occurrences (3)", "Diversity (2)", "Weighted frequency (4)",
                    "Number of weekends (13)", "Number of evenings (12)", "Co-occurrences weighted (5)", "Mutual co-occurrences (8)", "Specificity (14)"]
    else:
        X_train, y_train, X_test, y_test = fl.load_x_and_y()
        features = ["Two unique co-occurrences (0)", "Timely arrival and leaving (1)", "Diversity (2)", "Unique co-occurrences (3)", "Weighted frequency (4)",
                    "Co-occurrences weighted (5)", "Common travels (6)", "Number of co-occurrences (7)", "Mutual co-occurrences (8)", "Within 6 years of age (9)", "Same gender (10)",
                    "App usage similarity (11)", "Number of evenings (12)", "Number of weekends (13)", "Specificity (14)"]

    meets = X_train[np.where(y_train == 1)]
    nonmeets = X_train[np.where(y_train == 0)]
    figure = 1
    f = plt.figure(figure, figsize=(10, 10))
    plot_index = 1
    colors = ['#3498db', '#e74c3c']
    sns.set_palette(colors)
    for index, feature in enumerate(features):
        ax = f.add_subplot(2, 2, plot_index)
        ax.set_title(feature)
        all_events = np.hstack((meets[:, index], nonmeets[:, index]))
        types = np.hstack((["did meet" for x in range(len(meets[:, index]))], ["did not meet" for x in range(len(nonmeets[:,index]))]))
        if int(feature[-2]) not in [8]:
            if feature in ["Two unique co-occurrences (0)", "Within 6 years of age (9)", "Same gender (10)"]:
                sns.barplot(x=types, y=all_events, ci=False)
            else:
                sns.boxplot(x=types, y=all_events, ax=ax)
            [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
            ax.title.set_fontsize(48)
            [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
            plot_index += 1
            if plot_index == 5:
                plt.show()
                figure += 1
                f = plt.figure(figure, figsize=(10, 10))
                plot_index = 1
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]

    #f.tight_layout()
    plt.show()


if __name__ == '__main__':
    hist()