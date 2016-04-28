#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from FileLoader import FileLoader
from DatasetHelper import DatasetHelper
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.ensemble
import sklearn.svm
import sklearn.neighbors
from tqdm import tqdm

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
        Extract column 0 and 1 and removes dublicates row-wise

        Arguments:
            coocs {numpy array} -- Numpy array with at least 2 columns

        Returns:
            numpy array -- Numpy array with column 0 and 1 of input array.
                           Dublicates are removed
        """
        # Extract only column 0 & 1
        A = np.dstack((coocs[:, 0], coocs[:, 1]))[0]
        B = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize *
                                                   A.shape[1])))
        _, idx = np.unique(B, return_index=True) #Remove dublicate rows
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
            X[index:, 6] = datahelper.calculate_countries_in_common(user1, user2, loc_arr)
            X[index:, 7] = datahelper.calculate_number_of_common_travels(pair_coocs)
            y[index] = np.any(np.all([met_next[:, 0] == user1, met_next[:, 1] == user2], axis=0))

        return X, y

    def predict(self, X_train, y_train, X_test, y_test):
        print("Predicting baseline")
        lg = sklearn.linear_model.LogisticRegression()
        lg.fit(X_train[:, 0].reshape(-1, 1), y_train)
        y_pred = lg.predict(X_test[:, 0].reshape(-1, 1))
        print(lg.score(X_test[:, 0].reshape(-1, 1), y_test))
        print("Precision is: {}".format(sklearn.metrics.precision_score(y_test, y_pred)))
        print("Recall is: {}".format(sklearn.metrics.recall_score(y_test, y_pred)))
        cm_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print(cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis])


        forest = sklearn.ensemble.RandomForestClassifier()
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        print(forest.score(X_test, y_test))

        print("Precision is: {}".format(sklearn.metrics.precision_score(y_test, y_pred)))
        print("Recall is: {}".format(sklearn.metrics.recall_score(y_test, y_pred)))
        cm_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print(cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis])
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]


        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


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
    print(y_test.shape[0])
    print(y_train.shape[0])
    print("y_train contains {} that didnt meet, and {} that did meet".format(list(y_train).count(0), list(y_train).count(1)))
    print("y_test contains {} that didnt meet and {} that did meet".format(list(y_test).count(0), list(y_test).count(1)))
    p.predict(X_train, y_train, X_test, y_test)
