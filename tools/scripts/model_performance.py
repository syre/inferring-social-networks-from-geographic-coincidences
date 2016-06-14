#!/usr/bin/env python3
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import pickle
from scipy.stats import randint as sp_randint



def load_file(path_file):
    with open(path_file, "rb") as f:
        return pickle.load(f)


def two_unique(X_train, X_test):
    # create new feature two_unique_coocs from unique_coocs - production data only
    solo_feature_train = np.zeros(X_train[:, 1].shape)
    np.putmask(solo_feature_train, X_train[:, 1] >= 2, 1)
    solo_feature_test = np.zeros(X_test[:, 1].shape)
    np.putmask(solo_feature_test, X_test[:, 1] >= 2, 1) 
    return solo_feature_train, solo_feature_test


def gen_alldata():
    all_data = []
    #Test data
    X_train = load_file("../data/full_dataset/X_train.pickle")
    y_train = load_file("../data/full_dataset/y_train.pickle")
    X_test = load_file("../data/full_dataset/X_test.pickle")
    y_test = load_file("../data/full_dataset/y_test.pickle")

    solo_feature_train = X_train[:, 0]
    solo_feature_test = X_test[:, 0]
    np.delete(X_train, 0, 1)
    np.delete(X_test, 0, 1)
    all_data.append([(X_train, y_train, solo_feature_train, X_test, y_test,
                      solo_feature_test, "TP-1"),
                     (X_test, y_test, solo_feature_test, X_train, y_train,
                      solo_feature_train, "TP-1")])
    X_train = load_file("../data/reduced_dataset/X_train.pickle")
    y_train = load_file("../data/reduced_dataset/y_train.pickle")
    X_test = load_file("../data/reduced_dataset/X_test.pickle")
    y_test = load_file("../data/reduced_dataset/y_test.pickle")

    solo_feature_train = X_train[:, 0]
    solo_feature_test = X_test[:, 0]
    np.delete(X_train, 0, 1)
    np.delete(X_test, 0, 1)
    all_data.append([(X_train, y_train, solo_feature_train, X_test, y_test,
                      solo_feature_test, "TP-2"),
                     (X_test, y_test, solo_feature_test, X_train, y_train,
                      solo_feature_train, "TP-2")])
    #Production data
    X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_nonfilter_merged")
    X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_nonfilter_merged")
    X_train = X_train.toarray()
    X_test = X_test.toarray() 
    solo_feature_train, solo_feature_test = two_unique(X_train, X_test)
    all_data.append([(X_train, y_train, solo_feature_train, X_test, y_test,
                      solo_feature_test, "PP-1"),
                     (X_test, y_test, solo_feature_test, X_train, y_train,
                      solo_feature_train, "PP-1")])       
    X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
    X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")
    X_train = X_train.toarray()
    X_test = X_test.toarray() 
    solo_feature_train, solo_feature_test = two_unique(X_train, X_test)
    all_data.append([(X_train, y_train, solo_feature_train, X_test, y_test,
                      solo_feature_test, "PP-2"),
                     (X_test, y_test, solo_feature_test, X_train, y_train,
                      solo_feature_train, "PP-2")])

    return all_data


#[(X_train, y_train, solo_feature_train, X_test, y_test, solo_feature_test),
# (X_test, y_test, solo_feature_test, X_train, y_train, solo_feature_train)]
def plot_performance(all_data):
    param_grid = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 10),
                  "min_samples_split":  sp_randint(1, 10),#[1, 3, 10],
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "random_state": [0],
                  "class_weight": ["balanced", None]
                  }
    for i, pair in enumerate(all_data, start=1):
        ax = plt.subplot(2, 2, i)
        print("plot number {}".format(i))
        mean_tpr_lr = 0.0
        mean_fpr_lr = np.linspace(0, 1, 100)
        mean_tpr_rf = 0.0
        mean_fpr_rf = np.linspace(0, 1, 100)
        pair_name = ""
        for data in pair:
            pair_name = data[6]
           
            lr = sklearn.linear_model.LogisticRegression(random_state=0, class_weight="balanced")
            lr.fit(data[2].reshape(-1, 1), data[1])
            y_pred = lr.predict_proba(data[5].reshape(-1, 1))[:, 1]
            # logistic regression ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(data[4], y_pred)
            mean_tpr_lr += interp(mean_fpr_lr, false_positive_rate, true_positive_rate)

            forest = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
            grid_search = RandomizedSearchCV(forest, param_distributions=param_grid, scoring="roc_auc", n_jobs=-1, n_iter=20)
            grid_search.fit(data[0], data[1])
            print(grid_search.best_params_)
            y_pred = grid_search.predict_proba(data[3])[:, 1]
            # forest ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data[4], y_pred)
            mean_tpr_rf += interp(mean_fpr_rf, false_positive_rate, true_positive_rate)

        mean_tpr_lr /= 2
        mean_tpr_rf /= 2

        mean_tpr_lr[-1] = 1.0
        mean_tpr_rf[-1] = 1.0

        mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)

        mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)

        # Compute ROC curve and ROC area for each class
        plt.title('Receiver Operating Characteristic for ' + pair_name)
        plt.plot(mean_fpr_lr, mean_tpr_lr, 'r',
                 label='LR, Mean AUC = %0.2f' % mean_auc_lr)
        plt.plot(mean_fpr_rf, mean_tpr_rf, 'b',
                 label='RF, Mean AUC = %0.2f' % mean_auc_rf)

        plt.legend(loc='lower right', prop={'size': 40})
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #sns.plt.tick_params(labelsize=20)
        [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
        ax.title.set_fontsize(48)
        [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()


if __name__ == '__main__':
    plot_performance(gen_alldata())

#result for randomsearch with the following:
#----------------------------------------
#param_grid = {"max_depth": [3, None],
                  # "max_features": [1, 3, 10],
                  # "min_samples_split":  [1, 3, 10],
                  # "min_samples_leaf": [1, 3, 10],
                  # "bootstrap": [True, False],
                  # "criterion": ["gini", "entropy"],
                  # "random_state": [0],
                  # "class_weight": ["balanced", None]
                  # }
#lr = sklearn.linear_model.LogisticRegression(random_state=0, class_weight="balanced")
#forest = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
#grid_search = RandomizedSearchCV(forest, param_distributions=param_grid, scoring="roc_auc", n_jobs=-1, n_iter=20)
#-------------------------------------------
# plot number 1
# {'criterion': 'entropy', 'max_features': 3, 'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 3, 'random_state': 0, 'min_samples_leaf': 10}
# {'criterion': 'gini', 'max_features': 3, 'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 1, 'random_state': 0, 'min_samples_leaf': 1}
# plot number 2
# {'criterion': 'entropy', 'max_features': 1, 'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 3, 'min_samples_split': 3, 'random_state': 0, 'min_samples_leaf': 1}
# {'criterion': 'gini', 'max_features': 9, 'bootstrap': False, 'class_weight': None, 'max_depth': 3, 'min_samples_split': 10, 'random_state': 0, 'min_samples_leaf': 10}
# plot number 3
# {'criterion': 'entropy', 'max_features': 3, 'bootstrap': False, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 3, 'random_state': 0, 'min_samples_leaf': 3}
# {'criterion': 'entropy', 'max_features': 9, 'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 1, 'random_state': 0, 'min_samples_leaf': 3}
# plot number 4
# {'criterion': 'entropy', 'max_features': 9, 'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 10, 'random_state': 0, 'min_samples_leaf': 1}
# {'criterion': 'gini', 'max_features': 3, 'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 10, 'random_state': 0, 'min_samples_leaf': 1}
