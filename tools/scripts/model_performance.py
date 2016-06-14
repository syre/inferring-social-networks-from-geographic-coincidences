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
from sklearn.grid_search import GridSearchCV


X_train, y_train = load_svmlight_file("../data/vedran_thesis_students/X_train_filter_merged")
X_test, y_test = load_svmlight_file("../data/vedran_thesis_students/X_test_filter_merged")

X_train = X_train.toarray()
X_test = X_test.toarray()

# create new feature two_unique_coocs from unique_coocs
solo_feature_train = np.zeros(X_train[:, 1].shape)

np.putmask(solo_feature_train, X_train[:, 1] >= 2, 1)
solo_feature_test = np.zeros(X_test[:, 1].shape)
np.putmask(solo_feature_test, X_test[:, 1] >= 2, 1)

mean_tpr_lr = 0.0
mean_fpr_lr = np.linspace(0, 1, 100)

mean_tpr_rf = 0.0
mean_fpr_rf = np.linspace(0, 1, 100)

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 9],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "random_state": [0],
              "class_weight": ["balanced", None]
              }

for data in [(X_train, y_train, solo_feature_train, X_test, y_test, solo_feature_test),
             (X_test, y_test, solo_feature_test, X_train, y_train, solo_feature_train)]:
    lr = sklearn.linear_model.LogisticRegression(random_state=0, class_weight="balanced")
    lr.fit(data[2].reshape(-1, 1), data[1])
    y_pred = lr.predict_proba(data[5].reshape(-1, 1))[:, 1]
    # logistic regression ROC
    false_positive_rate, true_positive_rate, _ = roc_curve(
        data[4], y_pred)
    mean_tpr_lr += interp(mean_fpr_lr, false_positive_rate, true_positive_rate)

    forest = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
    grid_search = GridSearchCV(forest, param_grid=param_grid, scoring="roc_auc")
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
plt.title('Receiver Operating Characteristic for PP-2')
plt.plot(mean_fpr_lr, mean_tpr_lr, 'r',
         label='LR, Mean AUC = %0.2f' % mean_auc_lr)
plt.plot(mean_fpr_rf, mean_tpr_rf, 'b',
         label='RF, Mean AUC = %0.2f' % mean_auc_rf)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()