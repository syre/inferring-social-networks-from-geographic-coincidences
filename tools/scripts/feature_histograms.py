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
import pickle

def load_x_and_y(path):
    with open(os.path.join(path, "X_train.pickle"), "rb") as fp:
        X_train = pickle.load(fp)
    with open(os.path.join(path, "y_train.pickle"), "rb") as fp:
        y_train = pickle.load(fp)
    with open(os.path.join(path, "X_test.pickle"), "rb") as fp:
        X_test = pickle.load(fp)
    with open(os.path.join(path, "y_test.pickle"), "rb") as fp:
        y_test = pickle.load(fp)
    return X_train, y_train, X_test, y_test




def hist(is_prod=True):
    if is_prod:
        #[7, 3, 2, 4, 11, 10, 5, 8, 12]
        X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
        X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        features = ["Number of co-occurrences (7)", "Unique co-occurrences (3)", "Diversity (2)", "Weighted frequency (4)",
                    "Number of weekends (11)", "Number of evenings (10)", "Co-occurrences weighted (5)", "Mutual co-occurrences (8)", "Specificity (12)"]
    else:
        print("Test data"   )
        X_train, y_train, X_test, y_test = load_x_and_y("../data/reduced_dataset/")
        features = ["Two unique co-occurrences (0)", "Timely arrival and leaving (1)", "Diversity (2)", "Unique co-occurrences (3)", "Weighted frequency (4)",
                    "Co-occurrences weighted (5)", "Common travels (6)", "Number of co-occurrences (7)", "Mutual co-occurrences (8)",
                    "App usage similarity (9)", "Number of evenings (10)", "Number of weekends (11)", "Specificity (12)",
                    "Not within 6 years of age (13)", "Within 6 years of age (14)", "Within 6 years of age - Unknown (15)",
                    "Not same gender (16)", "Same gender (17)", "Same gender - Unknown (18)"]

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
        if feature not in ["Mutual co-occurrences (8)"]:
            if feature in ["Two unique co-occurrences (0)", "Not within 6 years of age (13)", "Within 6 years of age (14)", "Within 6 years of age - Unknown (15)",
                           "Not same gender (16)", "Same gender (17)", "Same gender - Unknown (18)"]:
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
    hist(False)