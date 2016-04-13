#!/usr/bin/env python3
from tqdm import tqdm
import json
import os
import pickle


class FileLoader():
    def generate_data_from_json(self, filenames, callback_func, path=""):
        for file_name in tqdm(filenames):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data):
                callback_func(row)

    def generate_app_data_from_json(self, callback_func, path=""):
        filenames = ["all_app_201509.json", "all_app_201510.json", "all_app_201511.json"]
        return self.generate_data_from_json(filenames, callback_func, path)

    def generate_location_data_from_json(self, callback_func, path=""):
        filenames = ["all_app_201509.json", "all_app_201510.json", "all_app_201511.json"]
        return self.generate_data_from_json(filenames, callback_func, path)

    def save_friend_and_nonfriend_pairs(self, friend_pairs, nonfriend_pairs):
        with open("friend_pairs.pickle", "wb") as fp:
            pickle.dump(friend_pairs, fp)
        with open("nonfriend_pairs.pickle", "wb") as fp:
            pickle.dump(nonfriend_pairs, fp)

    def load_friend_and_nonfriend_pairs(self):
        with open("friend_pairs.pickle", "rb") as fp:
            friend_pairs = pickle.load(fp)
        with open("nonfriend_pairs.pickle", "rb") as fp:
            nonfriend_pairs = pickle.load(fp)

        return friend_pairs, nonfriend_pairs

    def save_x_and_y(self, x, y):
        with open("datasetX.pickle", "wb") as fp:
            pickle.dump(x, fp)
        with open("datasetY.pickle", "wb") as fp:
            pickle.dump(y, fp)

    def load_x_and_y(self):
        with open("datasetX.pickle", "rb") as fp:
            x = pickle.load(fp)
        with open("datasetY.pickle", "rb") as fp:
            y = pickle.load(fp)
        return x, y
