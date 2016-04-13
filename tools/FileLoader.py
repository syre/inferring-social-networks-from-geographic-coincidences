#!/usr/bin/env python3
from tqdm import tqdm
import json
import os
import pickle
import numpy as np


class FileLoader():

    def generate_data_from_json(self, filenames, callback_func, path=""):
        for file_name in tqdm(filenames):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data):
                callback_func(row)

    def generate_app_data_from_json(self, callback_func, path=""):
        filenames = [
            "all_app_201509.json", "all_app_201510.json", "all_app_201511.json"]
        return self.generate_data_from_json(filenames, callback_func, path)

    def generate_location_data_from_json(self, callback_func, path=""):
        filenames = [
            "all_app_201509.json", "all_app_201510.json", "all_app_201511.json"]
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

    def load_numpy_matrix(self):
        with open("pickled_users.pickle", "rb") as f:
            users = pickle.load(f)

        with open("pickled_countries.pickle", "rb") as f:
            countries = pickle.load(f)

        with open("pickled_locations.npy", "rb") as f:
            numpy_arr = np.load(f)
        return users, countries, numpy_arr

    def generate_numpy_matrix_from_json(self, path=""):
        file_names = ["all_201509.json", "all_201510.json", "all_201511.json"]
        useruuid_dict = {}
        country_dict = {}

        user_count = 0
        country_count = 0
        locations = []
        for file_name in tqdm(file_names):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data, nested=True):
                spatial_bin = self.calculate_spatial_bin(
                    row["longitude"], row["latitude"])
                time_bins = self.calculate_time_bins(
                    row["start_time"], row["end_time"])
                for time_bin in time_bins:
                    if row["useruuid"] not in useruuid_dict:
                        user_count += 1
                        useruuid_dict[row["useruuid"]] = user_count
                    if row["country"] not in country_dict:
                        country_count += 1
                        country_dict[row["country"]] = country_count
                    useruuid = useruuid_dict[row["useruuid"]]
                    country = country_dict[row["country"]]

                    locations.append(
                        [useruuid, spatial_bin, time_bin, country])
        locations = np.array(locations)
        with open("pickled_users.pickle", "wb") as f:
            pickle.dump(useruuid_dict, f)
        with open("pickled_countries.pickle", "wb") as f:
            pickle.dump(country_dict, f)
        with open("pickled_locations.npy", "wb") as f:
            np.save(f, locations)

        return locations
