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
    coocs = dataset_helper.generate_cooccurrences_array(locations)
    file_loader.save_cooccurrences(coocs)
    print("processing friends and nonfriend pairs")
    friends, nonfriends = predictor.find_friend_and_nonfriend_pairs()
    file_loader.save_friend_and_nonfriend_pairs(friends, nonfriends)
    print("processing dataset for machine learning")
    X, y = predictor.generate_dataset(friends, nonfriends)
    file_loader.save_x_and_y(X, y)

if __name__ == '__main__':
    update_all_data()
