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
from sklearn.cross_validation import StratifiedKFold
import pickle
from scipy.stats import randint as sp_randint
import seaborn
plt.style.use("seaborn-deep")


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


def oversample(X_train, y_train):
    train_stacked = np.hstack((X_train, y_train.reshape(-1, 1)))
    didnt_meets = train_stacked[train_stacked[:,-1] == 0]
    did_meets = train_stacked[train_stacked[:,-1] == 1]
    train_stacked = np.vstack((didnt_meets[np.random.choice(didnt_meets.shape[0], did_meets.shape[0], replace=True)],
                               did_meets[np.random.choice(did_meets.shape[0], did_meets.shape[0], replace=False)]))
    y_train = train_stacked[:, -1]
    X_train = np.delete(train_stacked, -1, 1)
    return X_train, y_train


def undersample(X_train, y_train):
    train_stacked = np.hstack((X_train, y_train.reshape(-1, 1)))
    didnt_meets = train_stacked[train_stacked[:,-1] == 0]
    did_meets = train_stacked[train_stacked[:,-1] == 1]
    if didnt_meets.shape[0] > did_meets.shape[0]:
        train_stacked = np.vstack((didnt_meets[np.random.choice(didnt_meets.shape[0], did_meets.shape[0], replace=False)],
                                   did_meets[np.random.choice(did_meets.shape[0],
                                             did_meets.shape[0], replace=False)]))
    elif did_meets.shape[0] > didnt_meets.shape[0]:
        train_stacked = np.vstack((didnt_meets[np.random.choice(didnt_meets.shape[0], didnt_meets.shape[0], replace=False)],
                                   did_meets[np.random.choice(did_meets.shape[0],
                                             didnt_meets.shape[0], replace=False)]))
    y_train = train_stacked[:, -1]
    X_train = np.delete(train_stacked, -1, 1)
    return X_train, y_train
#[(X_train, y_train, solo_feature_train, X_test, y_test, solo_feature_test),
# (X_test, y_test, solo_feature_test, X_train, y_train, solo_feature_train)]
def plot_performance(all_data, undersampling=False):
    param_grid = {"max_features": sp_randint(1, 10),
                  "criterion": ["gini", "entropy"]
                  }
    importances = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i, pair in enumerate(all_data, start=1):
        ax = plt.subplot(2, 2, i)
        print("plot number {}".format(i))
        mean_tpr_lr = 0.0
        mean_fpr_lr = np.linspace(0, 1, 100)
        mean_tpr_rf = 0.0
        mean_fpr_rf = np.linspace(0, 1, 100)
        pair_name = ""
        precision_score_lr = [0,0]
        precision_score_rf = [0,0]
        recall_score_lr = [0,0]
        recall_score_rf = [0,0]
        for j, data in enumerate(pair):
            pair_name = data[6]

            lr = sklearn.linear_model.LogisticRegression(random_state=0, class_weight="balanced")
            lr.fit(data[2].reshape(-1, 1), data[1])
            y_pred = lr.predict_proba(data[5].reshape(-1, 1))[:, 1]
            precision_score_lr[0] += sklearn.metrics.precision_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=0)
            recall_score_lr[0] += sklearn.metrics.recall_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=0)
            precision_score_lr[1] += sklearn.metrics.precision_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=1)
            recall_score_lr[1] += sklearn.metrics.recall_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=1)
            # logistic regression ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(data[4], y_pred)
            mean_tpr_lr += interp(mean_fpr_lr, false_positive_rate, true_positive_rate)
            forest = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
            grid_search = RandomizedSearchCV(forest,
                                             param_distributions=param_grid,
                                             scoring="roc_auc", n_jobs=-1,
                                             n_iter=20)
            if undersampling:
                temp_data0, temp_data1 = undersample(data[0], data[1])
                grid_search.fit(temp_data0, temp_data1)
            else:
                grid_search.fit(data[0], data[1])
            print(grid_search.best_params_)
            y_pred = grid_search.predict_proba(data[3])[:, 1]
            precision_score_rf[0] += sklearn.metrics.precision_score(data[4], grid_search.predict(data[3]), pos_label=0)
            recall_score_rf[0] += sklearn.metrics.recall_score(data[4], grid_search.predict(data[3]), pos_label=0)
            precision_score_rf[1] += sklearn.metrics.precision_score(data[4], grid_search.predict(data[3]), pos_label=1)
            recall_score_rf[1] += sklearn.metrics.recall_score(data[4], grid_search.predict(data[3]), pos_label=1)
            # forest ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data[4], y_pred)
            mean_tpr_rf += interp(mean_fpr_rf, false_positive_rate, true_positive_rate)
            if pair_name == "PP-2":
                importances = np.add(grid_search.best_estimator_.feature_importances_, importances)
        #if i == 3 and j == 1:
        #    print(importances/2)
        mean_tpr_lr /= 2
        mean_tpr_rf /= 2

        mean_tpr_lr[-1] = 1.0
        mean_tpr_rf[-1] = 1.0

        mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)

        mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)
        print("For pair: {}\n---------".format(pair_name))
        print("Precision,Recall score (LR) for negative: {},{} and positive: {},{}".format(precision_score_lr[0]/2, recall_score_lr[0]/2, precision_score_lr[1]/2, recall_score_lr[1]/2))
        print("Precision,Recall score (RF) for negative: {},{} and positive: {},{}".format(precision_score_rf[0]/2, recall_score_rf[0]/2, precision_score_rf[1]/2, recall_score_rf[1]/2))
        print("---------")
        # Compute ROC curve and ROC area for each class
        if undersampling:
            plt.title('ROC for ' + pair_name + " with undersampling")
        else:
            plt.title('ROC for ' + pair_name)
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
    return importances/2


def plot_feature_importance(feature_impor, feature_id, undersampling):
    feature_impor, feature_id = [list(t) for t in zip(*sorted(zip(feature_impor,
                                                                  feature_id),
                                                              reverse=True))]
    print(feature_impor)
    ax = plt.subplot(1, 1, 1)
    if undersampling:
        plt.title("Feature importances of PP-2 with undersampling")
    else:
        plt.title("Feature importances of PP-2")
    plt.bar(range(9), feature_impor,
            color="#e74c3c", align="center")
    plt.xticks(range(9), feature_id)
    plt.xlabel("Feature id")
    plt.ylabel("Feature importance (%)")
    #plt.xlim([-1, 9])
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()

if __name__ == '__main__':
    undersampling = True
    feature_impor = plot_performance(gen_alldata(), undersampling=undersampling)
    feature_id = [7, 3, 2, 4, 13, 12, 5, 8, 14]
    plot_feature_importance(feature_impor, feature_id, undersampling=undersampling)


#num_coocs, num_unique_coocs, diversity, weighted_frequency, sum_weekends, sum_evenings, coocs_w, mutual_cooccurrences, specificity
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
