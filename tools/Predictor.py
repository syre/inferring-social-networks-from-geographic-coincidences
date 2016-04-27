#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from FileLoader import FileLoader
from DatasetHelper import DatasetHelper
import math
from dateutil import parser
import collections
import itertools
import numpy as np
import sklearn
import sklearn.ensemble
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
        X = np.empty(shape=(len(coocs_users), 5), dtype="float")
        y = np.empty(shape=(len(coocs_users), 1), dtype="int")

        for index, pair in tqdm(enumerate(coocs_users), total=coocs_users.shape[0]):
            user1 = pair[0]
            user2 = pair[1]

            pair_coocs = coocs[
                (coocs[:, 0] == user1) & (coocs[:, 1] == user2)]

            X[index:, 0] = datahelper.calculate_arr_leav(pair_coocs, loc_arr)
            X[index, 1] = datahelper.calculate_diversity(pair_coocs)
            X[index, 2] = datahelper.calculate_unique_cooccurrences(pair_coocs)
            X[index, 3] = datahelper.calculate_weighted_frequency(
                pair_coocs, loc_arr)
            X[index:, 4] = datahelper.calculate_coocs_w(pair_coocs, loc_arr)
            y[index] = np.any(np.all([[met_next[:, 0]] == user1,
                                      [met_next[:, 1]] == user2], axis=0))

        return X, y

    def predict(self, X_train, y_train, X_test, y_test):
        tree = sklearn.ensemble.RandomForestClassifier()
        tree.fit(X_train, y_train)
        print(tree.score(X_test, y_test))

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

    def calculate_specificity(self, user1, user2):
        """
        The asymmetric specificity Sij defined as fraction of time person pi spends with
        person pj with respect to the total time spent on campus by person pj . As
        shown in Table 4.3, the fraction of social time with respect to total time is more
        indicative of being perceived as a friend than only the social time.

        Feature ID: spec

        """
        pass

    
    # def find_friend_and_nonfriend_pairs(self, min_timebin, max_timebin):
    #     messaging_limit = 5*60
    #     phone_limit = 5
    #     phone_features = ["com.android.incallui"]
    #     messaging_features = ['com.snapchat.android',
    #                           'com.Slack',
    #                           'com.verizon.messaging.vzmsgs',
    #                           'jp.naver.line.android',
    #                           'com.whatsapp',
    #                           'org.telegram.messenger',
    #                           'com.google.android.talk',
    #                           'com.viber.voip',
    #                           'com.alibaba.android.rimet',
    #                           'com.skype.raider',
    #                           'com.sonyericsson.conversations',
    #                           'com.kakao.talk',
    #                           'com.google.android.apps.messaging',
    #                           'com.facebook.orca',
    #                           'com.tenthbit.juliet',
    #                           'com.tencent.mm']
    #     im_limits = {'com.snapchat.android': (),
    #                  'com.Slack': (),
    #                  'com.verizon.messaging.vzmsgs': (),
    #                  'jp.naver.line.android': (),
    #                  'com.whatsapp': (),
    #                  'org.telegram.messenger': (),
    #                  'com.google.android.talk': (),
    #                  'com.viber.voip': (),
    #                  'com.alibaba.android.rimet': (),
    #                  'com.skype.raider': (),
    #                  'com.sonyericsson.conversations': (),
    #                  'com.kakao.talk': (),
    #                  'com.google.android.apps.messaging': (),
    #                  'com.facebook.orca': (),
    #                  'com.tenthbit.juliet': (),
    #                  'com.tencent.mm': ()}

    #     user_info_dict = self.file_loader.generate_demographics_from_csv()
    #     user_info_dict = self.file_loader.filter_demographic_outliers(
    #         user_info_dict)
    #     rows = collections.defaultdict(list)
    #     country_users = self.database_helper.get_users_in_country(self.country)

    #     def callback_func(row):
    #         start_bins = self.database_helper.calculate_time_bins(
    #             row["start_time"], row["start_time"])
    #         end_bins = self.database_helper.calculate_time_bins(
    #             row["end_time"], row["end_time"])
    #         if row["package_name"] in phone_features+messaging_features and row["useruuid"] in country_users and start_bins[0] >= min_timebin and end_bins[0] < max_timebin:
    #             rows[row["useruuid"]].append(row)
    #     self.file_loader.generate_app_data_from_json(
    #         callback_func=callback_func)

    #     friend_pairs = []
    #     non_friend_pairs = []
    #     for pair in tqdm(list(itertools.combinations(rows.keys(), 2))):
    #         user1_records = rows[pair[0]]
    #         user2_records = rows[pair[1]]
    #         # coocs = self.database_helper.find_cooccurrences(pair[0],
    #         #                                                points_w_distances=self.database_helper.filter_places_dict[self.country],
    #         #                                                useruuid2=pair[1],
    #         #                                                min_timebin=min_timebin,
    #         #                                                max_timebin=max_timebin)

    #         if self.is_friends_on_app_usage(user1_records, user2_records, phone_features, phone_limit, messaging_features, messaging_limit) and self.is_friends_on_homophily(pair[0], pair[1], user_info_dict):
    #             friend_pairs.append((pair[0], pair[1]))
    #         else:
    #             non_friend_pairs.append((pair[0], pair[1]))

    #     return friend_pairs, non_friend_pairs

    # def is_friends_on_homophily(self, user_x, user_y, user_info_dict):
    #     if "age" not in user_info_dict[user_x] or "age" not in user_info_dict[user_y]:
    #         return False
    #     return abs(user_info_dict[user_x]["age"]-user_info_dict[user_y]["age"]) < 8

    # def is_friends_on_app_usage(self, user1_records, user2_records, phone_features, phone_limit, messaging_features, messaging_limit):
    #     for x in user1_records:
    #         start_time_x = parser.parse(x["start_time"])
    #         end_time_x = parser.parse(x["end_time"])
    #         for y in user2_records:
    #             if x["package_name"] != y["package_name"]:
    #                 continue
    #             start_time_y = parser.parse(y["start_time"])
    #             end_time_y = parser.parse(y["end_time"])
    #             start_diff = abs(start_time_x-start_time_y).seconds
    #             end_diff = abs(end_time_x-end_time_y).seconds

    #             if (x["package_name"] in messaging_features and start_diff < messaging_limit and end_diff < messaging_limit) or (x["package_name"] in phone_features and start_diff < phone_limit and end_diff < phone_limit):
    #                 return True
    #     else:
    #         return False


if __name__ == '__main__':
    p = Predictor("Japan")
    f = FileLoader()
    d = DatabaseHelper()
    print(
        len(list(itertools.combinations(d.get_users_in_country("Japan"), 2))))
    X_train, y_train, X_test, y_test = f.load_x_and_y()
    print(len(y_train[y_train == 1])+len(y_train[y_test == 1]),
          len(y_test[y_test == 0]) + len(y_train[y_train == 0]))
    p.predict(X_train, y_train, X_test, y_test)
