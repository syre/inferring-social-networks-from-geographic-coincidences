#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader

file_loader = FileLoader()
database_helper = DatabaseHelper()
predictor = Predictor()
dataset_helper = DatasetHelper()


class Run(object):
    """docstring for ClassName"""
    def __init__(self, train_dates_strings, test_dates_strings, country="Japan"):
        self.filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.train_dates = train_dates_strings
        self.test_dates = test_dates_strings
        self.country = country
        self.file_loader = FileLoader()
        self.database_helper = DatabaseHelper()
        self.predictor = Predictor(country=country,
                                   train_datetimes=train_dates_strings,
                                   test_datetimes=test_dates_strings)
        self.dataset_helper = DatasetHelper()


    def update_all_data(self):
        print("processing users, countries and locations as numpy matrix (train)")
        users_train, countries_train, locations_train = self.database_helper.generate_numpy_matrix_from_database()
        file_loader.save_numpy_matrix_train(users_train, countries_train, locations_train)

        print("processing users, countries and locations as numpy matrix (test)")
        users_test, countries_test, locations_test = self.database_helper.generate_numpy_matrix_from_database(self.filter_places_dict[self.country])
        file_loader.save_numpy_matrix_test(users_test, countries_test, locations_test)

        print("processing cooccurrences numpy array (train)")
        #coocs_train = dataset_helper.generate_cooccurrences_array(locations_train)
        #file_loader.save_cooccurrences_train(coocs_train)
        coocs_test = file_loader.load_cooccurrences_train()

        print("processing cooccurrences numpy array (test)")
        #coocs_test = dataset_helper.generate_cooccurrences_array(locations_train)
        #file_loader.save_cooccurrences_test(coocs_test)
        coocs_test = file_loader.load_cooccurrences_test()
        print("processing friends and nonfriend pairs (train)")

        sept_min_datetime = "2015-09-01 00:00:00+00:00"
        sept_min_time_bin = database_helper.calculate_time_bins(sept_min_datetime, sept_min_datetime)[0]
        sept_max_datetime = "2015-09-30 23:59:59+00:00"
        sept_max_time_bin = database_helper.calculate_time_bins(sept_max_datetime, sept_max_datetime)[0]
        oct_min_datetime = "2015-10-01 00:00:00+00:00"
        oct_min_time_bin = database_helper.calculate_time_bins(oct_min_datetime, oct_min_datetime)[0]
        oct_max_datetime = "2015-10-31 23:59:59+00:00"
        oct_max_time_bin = database_helper.calculate_time_bins(oct_max_datetime, oct_max_datetime)[0]
        nov_min_datetime = "2015-11-01 00:00:00+00:00"
        nov_min_time_bin = database_helper.calculate_time_bins(nov_min_datetime, nov_min_datetime)[0]
        nov_max_datetime = "2015-11-30 23:59:59+00:00"
        nov_max_time_bin = database_helper.calculate_time_bins(nov_max_datetime, nov_max_datetime)[0]

        train_friends, train_nonfriends = predictor.find_friend_and_nonfriend_pairs(sept_min_time_bin, oct_max_time_bin)
        print("processing dataset for machine learning (train)")
        X_train, y_train = predictor.generate_train_dataset()
        print("processing friends and nonfriend pairs (test)")
        test_friends, test_nonfriends = predictor.find_friend_and_nonfriend_pairs(nov_min_time_bin, nov_max_time_bin)
        print("processing dataset for machine learning (test)")
        X_test, y_test = predictor.generate_test_dataset()
        print("saving friends")
        file_loader.save_friend_and_nonfriend_pairs(train_friends, train_nonfriends, test_friends, test_nonfriends)
        print("saving dataset")
        file_loader.save_x_and_y(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predictor.predict(X_train, y_train, X_test, y_test)
if __name__ == '__main__':
    train_dates = ("2015-09-01 00:00:00+00:00", "2015-10-31 23:59:59+00:00")
    test_dates = ("2015-11-01 00:00:00+00:00", "2015-11-30 23:59:59+00:00")
    r = Run(train_dates_strings=train_dates, test_dates_strings=test_dates, country="Japan")
    r.update_all_data()
