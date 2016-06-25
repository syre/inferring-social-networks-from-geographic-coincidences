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


def filter_features(data_tuple, filter_list):
    prod = {7: 0, 3: 1, 2: 2,
            4: 3, 13: 4, 12: 5,
            8: 6, 14: 7}

    if data_tuple[6][:2] == "PP":
        filter_list = [prod[i] for i in filter_list]
    data_tuple[0] = data_tuple[0][:, filter_list]
    data_tuple[1] = data_tuple[1][:, filter_list]
    data_tuple[3] = data_tuple[3][:, filter_list]
    data_tuple[4] = data_tuple[4][:, filter_list]
    return data_tuple


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

    
def plot_performance(all_data, filter_feature_lst=[], undersampling=False,
                     target_importance="PP-2"):
    param_grid = {"max_features": sp_randint(1, 10),
                  "criterion": ["gini", "entropy"]
                  }
    if target_importance[0] == "T":
        importances = np.array([0]*19)
    else:
        importances = np.array([0]*9)
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
        conf_matrix_lr = np.array([[0,0],[0,0]])
        conf_matrix_rf = np.array([[0,0],[0,0]])
        for j, data in enumerate(pair):
            pair_name = data[6]
            if filter_feature_lst:
                data = filter_features(data, filter_feature_lst)
            lr = sklearn.linear_model.LogisticRegression(random_state=0, class_weight="balanced")
            lr.fit(data[2].reshape(-1, 1), data[1])
            y_pred = lr.predict_proba(data[5].reshape(-1, 1))[:, 1]
            precision_score_lr[0] += sklearn.metrics.precision_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=0)
            recall_score_lr[0] += sklearn.metrics.recall_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=0)
            precision_score_lr[1] += sklearn.metrics.precision_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=1)
            recall_score_lr[1] += sklearn.metrics.recall_score(data[4], lr.predict(data[5].reshape(-1, 1)), pos_label=1)
            conf_matrix_lr = np.add(conf_matrix_lr, confusion_matrix(data[4], lr.predict(data[5].reshape(-1, 1)), labels=[1, 0]))
            # logistic regression ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(data[4], y_pred)
            mean_tpr_lr += interp(mean_fpr_lr, false_positive_rate, true_positive_rate)
            forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
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
            conf_matrix_rf = np.add(conf_matrix_rf, confusion_matrix(data[4], grid_search.predict(data[3]), labels=[1, 0]))

            # forest ROC
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data[4], y_pred)
            mean_tpr_rf += interp(mean_fpr_rf, false_positive_rate, true_positive_rate)
            if pair_name == target_importance:
                importances = np.add(grid_search.best_estimator_.feature_importances_, importances)
        f_score_lr = (2 * (precision_score_lr[1]/2 * recall_score_lr[1]/2) / (precision_score_lr[1]/2 + recall_score_lr[1]/2))
        f_score_rf = (2 * (precision_score_rf[1]/2 * recall_score_rf[1]/2) / (precision_score_rf[1]/2 + recall_score_rf[1]/2))
        mean_tpr_lr /= 2
        mean_tpr_rf /= 2

        mean_tpr_lr[-1] = 1.0
        mean_tpr_rf[-1] = 1.0

        mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)

        mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)
        print("For pair: {}\n---------".format(pair_name))
        print("\t\tPrecision,\t\tRecall\t\t f-score \nLR:\t {}  \t{}  \t{}".format(precision_score_lr[1]/2, recall_score_lr[1]/2, f_score_lr))
        print("Confusion matrix LR : \n{}".format(conf_matrix_lr/2))
        print("RF:\t {}  \t{}  \t{}".format(precision_score_rf[1]/2, recall_score_rf[1]/2, f_score_rf))
        print("Confusion matrix RF : \n{}".format(conf_matrix_rf/2))
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
    return importances/2, target_importance


def plot_feature_importance(feature_impor, feature_id, pair_name, undersampling):
    feature_impor, feature_id = [list(t) for t in zip(*sorted(zip(feature_impor,
                                                                  feature_id),
                                                              reverse=True))]
    print(feature_impor)
    ax = plt.subplot(1, 1, 1)
    if undersampling:
        plt.title("Feature importances of " + pair_name + " with undersampling")
    else:
        plt.title("Feature importances of " + pair_name)
    plt.bar(range(len(feature_id)), feature_impor,
            color="#e74c3c", align="center")
    plt.xticks(range(len(feature_id)), feature_id)
    plt.xlabel("Feature id")
    plt.ylabel("Feature importance (%)")
    #plt.xlim([-1, 9])
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()

if __name__ == '__main__':
    undersampling = False
    target = "PP-2"
    feature_impor, pair = plot_performance(gen_alldata(), undersampling=undersampling, target_importance=target)
    #print("feature_impor = {}".format(feature_impor))
    if target[0] == "T":
        feature_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    else:
        feature_id = [7, 3, 2, 4, 11, 10, 5, 8, 12]

    plot_feature_importance(feature_impor, feature_id, pair,
                            undersampling=undersampling)