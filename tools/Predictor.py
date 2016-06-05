#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from FileLoader import FileLoader
from DatasetHelper import DatasetHelper
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import sklearn.ensemble
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from collections import Counter, defaultdict


class Predictor():

    def __init__(self,
                 country):
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
        country_arr = loc_arr[loc_arr[:, 3] == countries[self.country]]
        return country_arr

    def generate_dataset(self, users, countries, locations_arr, coocs,
                         met_next, min_timestring, max_timestring, selected_users=[]):
        min_timebin = self.database_helper.calculate_time_bins(
            min_timestring)[0]
        max_timebin = self.database_helper.calculate_time_bins(
            max_timestring)[0]
        # get only locations from specific country
        country_arr = self.filter_by_country(locations_arr, countries)

        # filter location array  and cooc array so its between max and min
        # timebin
        country_arr = locations_arr
        country_arr = country_arr[country_arr[:, 2] <= max_timebin]
        country_arr = country_arr[country_arr[:, 2] > min_timebin]
        coocs = coocs[coocs[:, 3] <= max_timebin]
        coocs = coocs[coocs[:, 3] > min_timebin]

        return self.calculate_features_for_dataset(users, countries,
                                                   country_arr, coocs,
                                                   met_next, min_timebin,
                                                   max_timebin, selected_users)

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
                                       met_next, min_timebin, max_timebin, selected_users=[]):
        datahelper = self.dataset_helper
        coocs_users = self.extract_and_remove_duplicate_coocs(coocs)
        if selected_users:
            coocs_users = coocs_users[np.in1d(coocs_users[:, 0], [users[u] for u in selected_users if u in users])]
            coocs_users = coocs_users[np.in1d(coocs_users[:, 1], [users[u] for u in selected_users if u in users])]
        X = np.zeros(shape=(len(coocs_users), 15), dtype="float")
        y = np.zeros(shape=len(coocs_users), dtype="int")
        demo_dict = self.file_loader.filter_demographic_outliers(self.file_loader.generate_demographics_from_csv())
        app_data_dict = defaultdict(set)

        def app_data_callback(row):
            # filter on min and max timebin
            start_bin = self.database_helper.calculate_time_bins(row["start_time"])[0]
            end_bin = self.database_helper.calculate_time_bins(row["end_time"])[0]
            if min_timebin <= start_bin and end_bin < max_timebin:
                app_data_dict[row["useruuid"]].add(row["package_name"])

        self.file_loader.generate_app_data_from_json(app_data_callback)
        for index, pair in tqdm(enumerate(coocs_users), total=coocs_users.shape[0]):
            user1 = pair[0]
            user2 = pair[1]

            pair_coocs = coocs[
                (coocs[:, 0] == user1) & (coocs[:, 1] == user2)]
            X[index:, 0] = self.has_two_unique_coocs(user1, user2, coocs)
            X[index:, 1] = datahelper.calculate_arr_leav(pair_coocs, loc_arr)
            X[index, 2] = datahelper.calculate_diversity(pair_coocs)
            X[index, 3] = datahelper.calculate_unique_cooccurrences(pair_coocs)
            X[index, 4] = datahelper.calculate_weighted_frequency(
                pair_coocs, loc_arr)
            X[index:, 5] = datahelper.calculate_coocs_w(pair_coocs, loc_arr)
            X[index:, 6] = datahelper.calculate_number_of_common_travels(
                pair_coocs)
            X[index:, 7] = pair_coocs.shape[0]
            X[index:, 8] = self.compute_mutual_cooccurrences(coocs, user1, user2)
            X[index:, 9] = self.is_within_6_years(demo_dict, users[user1], users[user2])
            X[index:, 10] = self.is_same_sex(demo_dict, users[user1], users[user2])
            X[index:, 11] = self.compute_app_jaccard_index(app_data_dict, users[user1], users[user2])
            X[index:, 12] = datahelper.calculate_number_of_evening_coocs(pair_coocs)
            X[index:, 13] = datahelper.calculate_number_of_weekend_coocs(pair_coocs)
            X[index:, 14] = datahelper.calculate_specificity(user1, user2, pair_coocs, loc_arr)
            y[index] = self.has_met(user1, user2, met_next)

        return X, y

    def compute_mutual_cooccurrences(self, coocs, user1, user2):
        user1_coocs = coocs[(coocs[:, 0] == user1) | (coocs[:, 1] == user1)]
        user2_coocs = coocs[(coocs[:, 0] == user2) | (coocs[:, 1] == user2)]
        user1_coocs = set(user1_coocs[:, 0]) | set(user1_coocs[:, 1])
        user2_coocs = set(user2_coocs[:, 0]) | set(user2_coocs[:, 1])
        return len(user1_coocs & user2_coocs)/len(user1_coocs | user2_coocs)

    def compute_app_jaccard_index(self, app_data_dict, user1, user2):
        union = len(app_data_dict[user1] | app_data_dict[user2])
        intersect = len(app_data_dict[user1] & app_data_dict[user2])
        if intersect == 0 and union == 0:
            return 1
        if union == 0:
            return 0
        return intersect/union

    def is_within_6_years(self, demo_dict, user1, user2):
        if user1 in demo_dict and user2 in demo_dict:
            diff = abs(demo_dict[user1]["age"]-demo_dict[user2]["age"])
            if diff <= 6:
                return 1
        return 0

    def is_same_sex(self, demo_dict, user1, user2):
        if user1 in demo_dict and user2 in demo_dict:
            if demo_dict[user1]["gender"] == demo_dict[user2]["gender"]:
                return 1
        return 0

    def has_met(self, user1, user2, met_next):
        return np.any(np.all([met_next[:, 0] == user1, met_next[:, 1] == user2], axis=0))

    def has_two_unique_coocs(self, user1, user2, coocs):
        tup = coocs.shape
        if len(tup) == 1:
            if tup[0] == 0:
                return 0
        else:
            if tup[0] == 1 and tup[1] == 0:
                return 0
        coocs = np.dstack((coocs[:, 0], coocs[:, 1],
                          coocs[:, 2]))[0]
        b = np.ascontiguousarray(coocs).view(
            np.dtype((np.void, coocs.dtype.itemsize * coocs.shape[1])))
        _, idx = np.unique(b, return_index=True)

        unique_met_next = coocs[idx]
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
        plt.xlabel("feature number")
        plt.ylabel("feature importance")
        plt.xlim([-1, X.shape[1]])
        plt.show()

    def compute_roc_curve(self, y_test, y_pred_proba):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, y_pred_proba)
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
        max_auc = (0,0,0)
        for x in range(1,500):
            for y in range(1,8):
                forest = sklearn.ensemble.RandomForestClassifier(n_estimators=x, class_weight="balanced", max_features=y)
                forest.fit(X_train, y_train)
                y_pred = forest.predict_proba(X_test)
                curr_auc = roc_auc_score(y_test, y_pred[:,1])
                if curr_auc > max_auc[0]:
                    max_auc = (curr_auc, x, y)
                    print("new max is: {}".format(max_auc))
        print("max is {} trees with auc of: {}".format(max_auc[0], max_auc[1]))

    def predict(self, X_train, y_train, X_test, y_test):
        scaler = StandardScaler().fit(np.vstack((X_train, X_test)))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print("Logistic Regression - with number of cooccurrences")
        lg = sklearn.linear_model.LogisticRegression(random_state=0)
        lg.fit(X_train[:, 0].reshape(-1, 1), y_train)
        y_pred = lg.predict(X_test[:, 0].reshape(-1, 1))
        print(sklearn.metrics.classification_report(y_test, y_pred))
        self.compute_roc_curve(y_test, lg.predict_proba(X_test[:, 0].reshape(-1, 1))[:,1])
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        self.plot_confusion_matrix(cm, title="Confusion matrix (Logistic Regression)")
        print(cm)
        print("Random Forest - all features")
        np.delete(X_train, [0], axis=1)
        np.delete(X_test, [0], axis=1)
        forest = sklearn.ensemble.RandomForestClassifier(n_estimators = 1000, random_state=0)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred))
        plt.style.use("ggplot")
        self.compute_feature_ranking(forest, X_test)
        plt.style.use("default")
        # compute ROC curve
        self.compute_roc_curve(y_test, forest.predict_proba(X_test)[:,1])
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, title="Confusion matrix (Random Forest)")
        print(cm)

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
    p = Predictor("Sweden")
    f = FileLoader()
    d = DatabaseHelper()
    X_train, y_train, X_test, y_test = f.load_x_and_y()
    #p.tweak_features(X_train, y_train, X_test, y_test)
    print("y_train contains {} that didnt meet, and {} that did meet".format(
        list(y_train).count(0), list(y_train).count(1)))
    print("y_test contains {} that didnt meet and {} that did meet".format(
        list(y_test).count(0), list(y_test).count(1)))
    p.predict(X_train, y_train, X_test, y_test)
