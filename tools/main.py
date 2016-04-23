#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader

file_loader = FileLoader()
database_helper = DatabaseHelper()
predictor = Predictor()
dataset_helper = DatasetHelper()


def update_all_data():
    print("processing users, countries and locations as numpy matrix")
    users, countries, locations = database_helper.generate_numpy_matrix_from_database()
    file_loader.save_numpy_matrix(users, countries, locations)
    print("processing cooccurrences numpy array")
    #coocs = dataset_helper.generate_cooccurrences_array(locations)
    #file_loader.save_cooccurrences(coocs)
    coocs = file_loader.load_cooccurrences()
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
    X_train, y_train = predictor.generate_dataset(train_friends, train_nonfriends, sept_min_time_bin, oct_max_time_bin)
    print("processing friends and nonfriend pairs (test)")
    test_friends, test_nonfriends = predictor.find_friend_and_nonfriend_pairs(nov_min_time_bin, nov_max_time_bin)
    print("processing dataset for machine learning (test)")
    X_test, y_test = predictor.generate_dataset(test_friends, test_nonfriends, nov_min_time_bin, nov_max_time_bin)
    print("saving friends")
    file_loader.save_friend_and_nonfriend_pairs(train_friends, train_nonfriends, test_friends, test_nonfriends)
    print("saving dataset")
    file_loader.save_x_and_y(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == '__main__':
    update_all_data()
