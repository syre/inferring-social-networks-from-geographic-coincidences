#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader
import numpy as np

file_loader = FileLoader()
database_helper = DatabaseHelper()
predictor = Predictor()
dataset_helper = DatasetHelper()


class Run(object):
    """docstring for ClassName"""
    def __init__(self, country="Japan"):
        self.filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.country = country
        self.file_loader = FileLoader()
        self.database_helper = DatabaseHelper()
        self.predictor = Predictor(country=country)
        self.dataset_helper = DatasetHelper()

    def update_all_data(self):
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

        print("processing users, countries and locations as numpy matrix (train)")
        users_train, countries_train, locations_train = self.database_helper.generate_numpy_matrix_from_database()
        file_loader.save_numpy_matrix_train(users_train, countries_train, locations_train)

        print("processing users, countries and locations as numpy matrix (test)")
        users_test, countries_test, locations_test = self.database_helper.generate_numpy_matrix_from_database(self.filter_places_dict[self.country])
        file_loader.save_numpy_matrix_test(users_test, countries_test, locations_test)

        print("processing cooccurrences numpy array (train)")
        coocs_train = dataset_helper.generate_cooccurrences_array(locations_train)
        file_loader.save_cooccurrences_train(coocs_train)
        coocs_train = file_loader.load_cooccurrences_train()

        print("processing cooccurrences numpy array (test)")
        coocs_test = dataset_helper.generate_cooccurrences_array(locations_test)
        file_loader.save_cooccurrences_test(coocs_test)
        coocs_test = file_loader.load_cooccurrences_test()

        print("processing coocs for met in next (train)")
        coocs_met_in_next_train = np.copy(coocs_train)
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] <= oct_max_time_bin]
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] > oct_min_time_bin]

        print("finding met in next people (train)")
        met_in_next_train = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_train)
        print("saving met in next people (train)")
        file_loader.save_met_in_next_train(met_in_next_train)

        print("processing coocs for met in next (test)")
        coocs_met_in_next_test = np.copy(coocs_test)
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] <= nov_max_time_bin]
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] > nov_min_time_bin]

        print("finding met in next people (test)")
        met_in_next_test = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_test)
        print("saving met in next people (test)")
        file_loader.save_met_in_next_test(met_in_next_test)

        print("processing dataset for machine learning (train)")
        X_train, y_train = predictor.generate_dataset(users_train, countries_train, locations_train, coocs_train, met_in_next_train, sept_min_datetime, sept_max_datetime)
        print("processing dataset for machine learning (test)")
        X_test, y_test = predictor.generate_dataset(users_test, countries_test, locations_test, coocs_test, met_in_next_test, oct_min_datetime, oct_max_datetime)
        print("saving dataset")
        file_loader.save_x_and_y(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predictor.predict(X_train, y_train, X_test, y_test)
if __name__ == '__main__':
    r = Run(country="Japan")
    r.update_all_data()
