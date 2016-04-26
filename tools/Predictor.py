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
import random


class Predictor():

    def __init__(self,
                 country="Japan",
                 train_datetimes=(
                     "2015-09-01 00:00:00+00:00", "2015-10-31 23:59:59+00:00"),
                 test_datetimes=("2015-11-01 00:00:00+00:00",
                                 "2015-11-30 23:59:59+00:00")):
        """
            Constructor

            Args:
                   country: country used for generating dataset and prediction
        """
        self.database_helper = DatabaseHelper()
        self.dataset_helper = DatasetHelper()
        self.file_loader = FileLoader()
        self.country = country
        self.train_datetimes = train_datetimes
        self.test_datetimes = test_datetimes

    def filter_by_country(self, loc_arr, countries):
        country_arr = loc_arr[
            np.in1d([loc_arr[:, 3]], [countries[self.country]])]
        return country_arr

    def generate_train_dataset(self, users, countries, locations_arr, coocs,
                               met_next):
        min_timebin = self.database_helper.calculate_time_bins(
            self.train_datetimes[0], self.train_datetimes[0])[0]
        max_timebin = self.database_helper.calculate_time_bins(
            self.train_datetimes[1], self.train_datetimes[1])[0]
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

    def generate_test_dataset(self, users, countries, locations_arr, coocs,
                              met_next):
        min_timebin = self.database_helper.calculate_time_bins(
            self.test_datetimes[0], self.test_datetimes[0])[0]
        max_timebin = self.database_helper.calculate_time_bins(
            self.test_datetimes[1], self.test_datetimes[1])[0]

        # get only locations from specific country
        country_arr = self.filter_by_country(locations_arr, countries)

        # filter location array  and cooc array so its between max and min
        # timebin
        country_arr = country_arr[country_arr[:, 2] <= max_timebin]
        country_arr = country_arr[country_arr[:, 2] > min_timebin]

        return self.calculate_features_for_dataset(users, countries,
                                                   country_arr, coocs,
                                                   met_next)

    def find_met_in_next_pairs(self, coocs):
        return np.dstack((coocs[:, 0], coocs[:, 1]))[0]

    def calculate_features_for_dataset(self, users, countries, loc_arr, coocs,
                                       met_next):
        datahelper = self.dataset_helper

        X = np.ndarray(shape=(len(coocs), 6), dtype="float")
        y = np.empty(shape=(len(coocs), 1), dtype="int")

        for index, pair in tqdm(enumerate(coocs)):
            user1 = users[pair[0]]
            user2 = users[pair[1]]

            pair1_coocs = coocs[
                (coocs[:, 0] == user1) & (coocs[:, 1] == user2)]
            pair2_coocs = coocs[
                (coocs[:, 0] == user2) & (coocs[:, 1] == user1)]
            pair_coocs = np.vstack((pair1_coocs, pair2_coocs))
            X[index:, 1] = datahelper.calculate_arr_leav(pair_coocs, loc_arr)
            X[index, 2] = datahelper.calculate_diversity(pair_coocs)
            X[index, 3] = datahelper.calculate_unique_cooccurrences(pair_coocs)
            X[index, 4] = datahelper.calculate_weighted_frequency(
                pair_coocs, loc_arr)
            X[index:, 5] = datahelper.calculate_coocs_w(pair_coocs, loc_arr)
            y[index] = np.any(np.all([met_next[:, 0] == user1,
                                      met_next[:, 1] == user2], axis=0)) or \
                np.any(np.all([met_next[:, 0] == user2,
                               met_next[:, 1] == user1], axis=0))

            #y[index] = (user1, user2) in met_next or (user2, user1) in met_next

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

    def calculate_arr_leav(self, cooccurrences):
        """
        We propose that if two persons arrive at a location at the same time and/or
        leave the location synchronously it yields a stronger signal than if two people
        are in the same location, but their arrival and leaving are not synchronized. The
        value is weighted by the number of people who arrived and/or left the building
        in each particular time bin. Thus, timed arrival of many people in the beginning
        of the scheduled classes is not as strong a signal as synchronized arrival of a few
        persons an hour before the class begins. 

        Feature ID: arr_leav
        """

        arr_leav_values = []
        if len(cooccurrences) == 0:
            print("no cooccurrences in arr_leav")
            return 0

        for cooc in cooccurrences:
            spatial_bin = cooc[1]
            arr_leav_value = 0

            # sort timebins to reliably get previous and next timebins outside
            # their cooc
            user1_time_bins = cooc[2]
            user2_time_bins = cooc[3]

            if not (min(user1_time_bins) == min(user2_time_bins)):
                # non-synchronously arrival
                arr_leav_value += 0
            else:
                # synchronous arrival
                before_arrive_list = self.find_users_in_cooccurrence(
                    spatial_bin, min(user1_time_bins)-1)
                arrive_list = self.find_users_in_cooccurrence(
                    spatial_bin, min(user1_time_bins))

                number_of_new_arrivals = len(
                    set(arrive_list)-set(before_arrive_list))

                if number_of_new_arrivals == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/(number_of_new_arrivals))

            if not (max(user1_time_bins) == max(user2_time_bins)):
                # non-synchronously leaving
                arr_leav_value += 0
            else:
                # synchronous leaving
                leave_list = self.find_users_in_cooccurrence(
                    spatial_bin, max(user1_time_bins))
                after_leave_list = self.find_users_in_cooccurrence(
                    spatial_bin, max(user1_time_bins)+1)

                number_of_leavers = len(set(after_leave_list)-set(leave_list))

                if number_of_leavers == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/(number_of_leavers))

            arr_leav_values.append(arr_leav_value)

        return sum(arr_leav_values)/len(cooccurrences)

    def calculate_coocs_w(self, cooccurrences):
        """
        While other researchers use entropy to weight the social impact of meetings, our
        data allows us to introduce a more precise measure. We use anonymous statistics
        to estimate the number of all people present in the building in each time bin. We
        assume the social importance of each co-occurrence to be inversely proportional
        to the number of people – if only a few persons are there in a location, it is more
        probable that there is a social bond between them compared to the situation
        when dozens of people are present.

        Calculates all values of coocs_w for cooccurrences and returns the mean of them 

        Feature ID: coocs_w
        """

        coocs_w_values = []
        if len(cooccurrences) == 0:
            return 0
        for cooc in cooccurrences:
            lng = cooc[1]
            lat = cooc[2]

            coocs_w_value = 0
            spatial_bin = cooc[1]
            user1_time_bins = cooc[2]
            user2_time_bins = cooc[3]
            common_time_bins = set(user1_time_bins) & set(user2_time_bins)
            for time_bin in common_time_bins:
                users = self.find_users_in_cooccurrence(spatial_bin, time_bin)
                num_users = len(users)

                if num_users < 2:
                    print(
                        "only {} together in timebin, coocs_w".format(str(num_users)))
                    continue
                coocs_w_value += num_users

            coocs_w_value /= len(common_time_bins)

            # 2 users is ideal thus returning highest value 1, else return
            # lesser value proportional to amount of users
            coocs_w_values.append(1/(coocs_w_value-1))

        return sum(coocs_w_values)/len(cooccurrences)

    def calculate_specificity(self, user1, user2):
        """
        The asymmetric specificity Sij defined as fraction of time person pi spends with
        person pj with respect to the total time spent on campus by person pj . As
        shown in Table 4.3, the fraction of social time with respect to total time is more
        indicative of being perceived as a friend than only the social time.

        Feature ID: spec

        """
        pass

    def calculate_diversity(self, cooccurrences):
        """
        Diversity quantifies how many locations the cooccurrences between two people represent.
        Can either use Shannon or Renyi Entropy, right now uses Shannon
        From inferring realworld relationships from spatiotemporal data paper p. 23.
        """

        frequency = len(cooccurrences)
        spatial_bins_counts = collections.Counter(
            [cooc[1] for cooc in cooccurrences])

        shannon_entropy = -sum([(count/frequency)*math.log(count/frequency, 2)
                                for _, count in spatial_bins_counts.items()])

        return np.exp(shannon_entropy)

    def calculate_weighted_frequency(self, cooccurrences):
        """
        Inferring realworld relationships from spatiotemporal data paper p. 19 and 23-25
        Tells how important co-occurrences are at non-crowded places
        """

        spatial_bins_counts = collections.Counter(
            [cooc[1] for cooc in cooccurrences])

        weighted_frequency = 0
        for spatial_bin, count in spatial_bins_counts.items():
            unique_users = self.database_helper.find_cooccurrences_within_area(
                spatial_bin)
            location_entropy = 0
            for user in unique_users:
                v_lu = self.database_helper.get_locations_for_user(
                    user, spatial_bin=spatial_bin)
                v_l = self.database_helper.find_number_of_records_for_location(
                    spatial_bin)
                prob = len(v_lu)/len(v_l)
                if prob != 0:
                    location_entropy += prob*math.log(prob, 2)
            location_entropy = -location_entropy
            weighted_frequency += count * np.exp(-location_entropy)
        return weighted_frequency

    def find_friend_and_nonfriend_pairs(self, min_timebin, max_timebin):
        messaging_limit = 5*60
        phone_limit = 5
        phone_features = ["com.android.incallui"]
        messaging_features = ['com.snapchat.android',
                              'com.Slack',
                              'com.verizon.messaging.vzmsgs',
                              'jp.naver.line.android',
                              'com.whatsapp',
                              'org.telegram.messenger',
                              'com.google.android.talk',
                              'com.viber.voip',
                              'com.alibaba.android.rimet',
                              'com.skype.raider',
                              'com.sonyericsson.conversations',
                              'com.kakao.talk',
                              'com.google.android.apps.messaging',
                              'com.facebook.orca',
                              'com.tenthbit.juliet',
                              'com.tencent.mm']
        im_limits = {'com.snapchat.android': (),
                     'com.Slack': (),
                     'com.verizon.messaging.vzmsgs': (),
                     'jp.naver.line.android': (),
                     'com.whatsapp': (),
                     'org.telegram.messenger': (),
                     'com.google.android.talk': (),
                     'com.viber.voip': (),
                     'com.alibaba.android.rimet': (),
                     'com.skype.raider': (),
                     'com.sonyericsson.conversations': (),
                     'com.kakao.talk': (),
                     'com.google.android.apps.messaging': (),
                     'com.facebook.orca': (),
                     'com.tenthbit.juliet': (),
                     'com.tencent.mm': ()}

        user_info_dict = self.file_loader.generate_demographics_from_csv()
        user_info_dict = self.file_loader.filter_demographic_outliers(
            user_info_dict)
        rows = collections.defaultdict(list)
        country_users = self.database_helper.get_users_in_country(self.country)

        def callback_func(row):
            start_bins = self.database_helper.calculate_time_bins(
                row["start_time"], row["start_time"])
            end_bins = self.database_helper.calculate_time_bins(
                row["end_time"], row["end_time"])
            if row["package_name"] in phone_features+messaging_features and row["useruuid"] in country_users and start_bins[0] >= min_timebin and end_bins[0] < max_timebin:
                rows[row["useruuid"]].append(row)
        self.file_loader.generate_app_data_from_json(
            callback_func=callback_func)

        friend_pairs = []
        non_friend_pairs = []
        for pair in tqdm(list(itertools.combinations(rows.keys(), 2))):
            user1_records = rows[pair[0]]
            user2_records = rows[pair[1]]
            # coocs = self.database_helper.find_cooccurrences(pair[0],
            #                                                points_w_distances=self.database_helper.filter_places_dict[self.country],
            #                                                useruuid2=pair[1],
            #                                                min_timebin=min_timebin,
            #                                                max_timebin=max_timebin)

            if self.is_friends_on_app_usage(user1_records, user2_records, phone_features, phone_limit, messaging_features, messaging_limit) and self.is_friends_on_homophily(pair[0], pair[1], user_info_dict):
                friend_pairs.append((pair[0], pair[1]))
            else:
                non_friend_pairs.append((pair[0], pair[1]))

        return friend_pairs, non_friend_pairs

    def is_friends_on_homophily(self, user_x, user_y, user_info_dict):
        if "age" not in user_info_dict[user_x] or "age" not in user_info_dict[user_y]:
            return False
        return abs(user_info_dict[user_x]["age"]-user_info_dict[user_y]["age"]) < 8

    def is_friends_on_app_usage(self, user1_records, user2_records, phone_features, phone_limit, messaging_features, messaging_limit):
        for x in user1_records:
            start_time_x = parser.parse(x["start_time"])
            end_time_x = parser.parse(x["end_time"])
            for y in user2_records:
                if x["package_name"] != y["package_name"]:
                    continue
                start_time_y = parser.parse(y["start_time"])
                end_time_y = parser.parse(y["end_time"])
                start_diff = abs(start_time_x-start_time_y).seconds
                end_diff = abs(end_time_x-end_time_y).seconds

                if (x["package_name"] in messaging_features and start_diff < messaging_limit and end_diff < messaging_limit) or (x["package_name"] in phone_features and start_diff < phone_limit and end_diff < phone_limit):
                    return True
        else:
            return False


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
