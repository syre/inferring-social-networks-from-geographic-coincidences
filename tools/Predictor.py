#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from FileLoader import FileLoader
from DatasetHelper import DatasetHelper
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn.ensemble
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

class Predictor():

    def __init__(self,
                 country="Japan"):
        """
            Constructor

            Args:
                   country: country used for generating dataset and prediction
        """
        self.database_helper = DatabaseHelper()
        self.dataset_helper = DatasetHelper()
        self.file_loader = FileLoader()
        self.country = country

    def filter_by_country(self, loc_arr, countries):
        country_arr = loc_arr[
            np.in1d([loc_arr[:, 3]], [countries[self.country]])]
        return country_arr

    def generate_dataset(self, users, countries, locations_arr, coocs,
                         met_next, min_timestring, max_timestring):
        min_timebin = self.database_helper.calculate_time_bins(
            min_timestring, min_timestring)[0]
        max_timebin = self.database_helper.calculate_time_bins(
            max_timestring, max_timestring)[0]
        # get only locations from specific country
        country_arr = self.filter_by_country(locations_arr, countries)

        # filter location array  and cooc array so its between max and min
        # timebin
        country_arr = country_arr[country_arr[:, 2] <= max_timebin]
        country_arr = country_arr[country_arr[:, 2] > min_timebin]
        coocs = coocs[coocs[:, 3] <= max_timebin]
        coocs = coocs[coocs[:, 3] > min_timebin]

        return self.calculate_features_for_dataset(users, countries,
                                                   country_arr, coocs,
                                                   met_next)

    def extract_and_remove_duplicate_coocs(self, coocs):
        """
        Extract column 0 and 1 and removes duplicates row-wise

        Arguments:
            coocs {numpy array} -- Numpy array with at least 2 columns

        Returns:
            numpy array -- Numpy array with column 0 and 1 of input array.
                           Duplicates are removed
        """
        # Extract only column 0 & 1
        A = np.dstack((coocs[:, 0], coocs[:, 1]))[0]
        B = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize *
                                                   A.shape[1])))
        _, idx = np.unique(B, return_index=True)  # Remove duplicate rows
        return A[idx]

    def calculate_features_for_dataset(self, users, countries, loc_arr, coocs,
                                       met_next):
        datahelper = self.dataset_helper
        coocs_users = self.extract_and_remove_duplicate_coocs(coocs)
        X = np.zeros(shape=(len(coocs_users), 8), dtype="float")
        y = np.zeros(shape=len(coocs_users), dtype="int")

        for index, pair in tqdm(enumerate(coocs_users), total=coocs_users.shape[0]):
            user1 = pair[0]
            user2 = pair[1]

            pair_coocs = coocs[
                (coocs[:, 0] == user1) & (coocs[:, 1] == user2)]
            X[index:, 0] = pair_coocs.shape[0]
            X[index:, 1] = datahelper.calculate_arr_leav(pair_coocs, loc_arr)
            X[index, 2] = datahelper.calculate_diversity(pair_coocs)
            X[index, 3] = datahelper.calculate_unique_cooccurrences(pair_coocs)
            X[index, 4] = datahelper.calculate_weighted_frequency(
                pair_coocs, loc_arr)
            X[index:, 5] = datahelper.calculate_coocs_w(pair_coocs, loc_arr)
            X[index:, 6] = datahelper.calculate_countries_in_common(
                user1, user2, loc_arr)
            X[index:, 7] = datahelper.calculate_number_of_common_travels(
                pair_coocs)
            y[index] = self.has_met(user1, user2, met_next)

        return X, y

    def has_met(self, user1, user2, met_next):
        return np.any(np.all([met_next[:, 0] == user1, met_next[:, 1] == user2], axis=0))

    def has_two_unique_coocs(self, user1, user2, met_next):
        tup = met_next.shape
        if len(tup) == 1:
            if tup[0] == 0:
                return 0
        else:
            if tup[0] == 1 and tup[1] == 0:
                return 0
        met_next = np.dstack((met_next[:, 0], met_next[:, 1],
                              met_next[:, 2]))[0]
        b = np.ascontiguousarray(met_next).view(
            np.dtype((np.void, met_next.dtype.itemsize * met_next.shape[1])))
        _, idx = np.unique(b, return_index=True)

        unique_met_next = met_next[idx]
        unique_pair_rows = unique_met_next[np.all([unique_met_next[:, 0] ==
                                                   user1,
                                                   unique_met_next[:, 1] ==
                                                   user2], axis=0)]
        return unique_pair_rows.shape[0] >= 2
    def compute_feature_ranking(self, forest, X):
        importances = forest.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    def compute_roc_curve(self, y_test, y_pred_proba):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_test, y_pred_proba, pos_label=1)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # Compute ROC curve and ROC area for each class
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_confusion_matrix(self, cm, title="Confusion matrix", target_names=["did not meet", "did meet"], cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def tweak_features(self, X_train, y_train, X_test, y_test):
        max_auc = (0,0)
        for x in range(1,200):
            forest = sklearn.ensemble.RandomForestClassifier(n_estimators=x)
            forest.fit(X_train, y_train)
            y_pred = forest.predict_proba(X_test)
            curr_auc = roc_auc_score(y_test, y_pred[:,1])
            if curr_auc > max_auc[0]:
                max_auc = (curr_auc, x)
                print("new max is: {}".format(max_auc))
        print("max is {} trees with auc of: {}".format(max_auc[0], max_auc[1]))

    def predict(self, X_train, y_train, X_test, y_test):
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        print("Logistic Regression - with number of cooccurrences")
        lg = sklearn.linear_model.LogisticRegression()
        lg.fit(X_train[:, 0].reshape(-1, 1), y_train)
        y_pred = lg.predict(X_test[:, 0].reshape(-1, 1))
        print(sklearn.metrics.classification_report(y_test, y_pred, target_names=["didnt meet", "did meet"]))
        self.compute_roc_curve(y_test, lg.predict_proba(X_test[:, 0].reshape(-1, 1))[:,1])
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)

        print("Random Forest - all features")
        #self.tweak_features(X_train, y_train, X_test, y_test)
        forest = sklearn.ensemble.RandomForestClassifier()
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred, target_names=["didnt meet", "did meet"]))
        self.compute_feature_ranking(forest, X_test)
        # compute ROC curve
        self.compute_roc_curve(y_test, forest.predict_proba(X_test)[:,1])
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)

    def predict2(self, X_train, y_train, X_test, y_test):
        print("Logistic Regression - with number of cooccurrences")

        lg = sklearn.linear_model.LogisticRegression()
        param_grid = {"class_weight": {1: 2},
                      "C": np.arange(0, 1, 0.1)} #did_meet weight = 2
        grid = GridSearchCV(estimator=lg, param_grid=dict(alpha=alphas))
        lg.fit(X_train[:, 0].reshape(-1, 1), y_train)
        y_pred = lg.predict(X_test[:, 0].reshape(-1, 1))
        print(sklearn.metrics.classification_report(y_pred, y_test, target_names=["didnt meet", "did meet"]))
        #self.compute_roc_curve(y_test, y_pred)
        #cm = confusion_matrix(y_test, y_pred)
        #self.plot_confusion_matrix(cm)

        print("Random Forest - all features")
        forest = sklearn.ensemble.RandomForestClassifier()
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        print(sklearn.metrics.classification_report(y_pred, y_test, target_names=["didnt meet", "did meet"]))
        self.compute_feature_ranking(forest, X_test)
        # compute ROC curve
        self.compute_roc_curve(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)

    def find_users_in_cooccurrence(self, spatial_bin, time_bin):
        """
        Find all users who's been in a given cooccurrence
            Arguments:
                lat {float} -- latitude
                lng {float} -- longitude
                time_bin {integer} -- time bin index
            Returns:
                list -- list of user_uuids       
        """
        return self.database_helper.find_cooccurrences_within_area(spatial_bin,
                                                                   time_bin)

    def calculate_unique_cooccurrences(self, cooccurrences):
        """
        Calculates how many unique spatial bins they have had cooccurrences in
        """

        return len(set([cooc[1] for cooc in cooccurrences]))


if __name__ == '__main__':
    p = Predictor("Japan")
    f = FileLoader()
    d = DatabaseHelper()
    X_train, y_train, X_test, y_test = f.load_x_and_y()
    print("y_train contains {} that didnt meet, and {} that did meet".format(
        list(y_train).count(0), list(y_train).count(1)))
    print("y_test contains {} that didnt meet and {} that did meet".format(
        list(y_test).count(0), list(y_test).count(1)))
    p.predict(X_train, y_train, X_test, y_test)
