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
    users_for_delete = ['02cbb276-93e7-4baa-81aa-ea3f5bf6230a',
                        '3eee0fe1-ef56-42a9-9b16-ee0677c079ee',
                        '578669a8-4b85-49a2-bf46-5437d6192252',
                        '7cb9fddd-d3b6-49f8-9259-51b948c2ac1f',
                        '8298d998-684a-4e90-9a1e-cd0164dabf2e',
                        'e8d40f3a-1e07-4c26-adfb-39a8366d4bbd',
                        'ebb04181-3cc2-4fac-a34f-1962a6953081']
    database_helper.delete_users(users_for_delete)
    update_all_data()
