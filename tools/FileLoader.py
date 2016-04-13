#!/usr/bin/env python3
from tqdm import tqdm
import json
import os
import pickle
import numpy as np
from collections import defaultdict
from dateutil import parser
from datetime import datetime
import csv


class FileLoader():

    def generate_data_from_json(self, filenames, callback_func, path="data"):
        for file_name in tqdm(filenames):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data):
                callback_func(row)

    def generate_demographics_from_csv(self, path="data"):
        file_name = "user_data.csv"
        user_info_dict = defaultdict(list)

        with open(os.path.join(path, file_name)) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                birthdate = parser.parse(row[2])
                age = datetime.now()-birthdate
                user_info_dict[row[0]] = {"gender": row[1], "age": age.days/365}
        return user_info_dict

    def generate_app_data_from_json(self, callback_func, path="data"):
        filenames = [
            "all_app_201509.json", "all_app_201510.json", "all_app_201511.json"]
        return self.generate_data_from_json(filenames, callback_func, path)

    def generate_location_data_from_json(self, callback_func, path="data"):
        filenames = [
            "all_201509.json", "all_201510.json", "all_201511.json"]
        return self.generate_data_from_json(filenames, callback_func, path)

    def save_friend_and_nonfriend_pairs(self, friend_pairs, nonfriend_pairs):
        with open(os.path.join("data", "friend_pairs.pickle"), "wb") as fp:
            pickle.dump(friend_pairs, fp)
        with open(os.path.join("data", "nonfriend_pairs.pickle"), "wb") as fp:
            pickle.dump(nonfriend_pairs, fp)

    def load_friend_and_nonfriend_pairs(self):
        with open(os.path.join("data", "friend_pairs.pickle"), "rb") as fp:
            friend_pairs = pickle.load(fp)
        with open(os.path.join("data", "nonfriend_pairs.pickle"), "rb") as fp:
            nonfriend_pairs = pickle.load(fp)

        return friend_pairs, nonfriend_pairs

    def load_cooccurrences(self):
        with open(os.path.join("data", "cooccurrences.npy"), "rb") as f:
            coocs = np.load(f)
        return coocs

    def save_x_and_y(self, x, y):
        with open(os.path.join("data", "datasetX.pickle"), "wb") as fp:
            pickle.dump(x, fp)
        with open(os.path.join("data", "datasetY.pickle"), "wb") as fp:
            pickle.dump(y, fp)

    def load_x_and_y(self):
        with open(os.path.join("data", "datasetX.pickle"), "rb") as fp:
            x = pickle.load(fp)
        with open(os.path.join("data", "datasetY.pickle"), "rb") as fp:
            y = pickle.load(fp)
        return x, y

    def load_numpy_matrix(self):
        with open(os.path.join("data", "pickled_users.pickle"), "rb") as f:
            users = pickle.load(f)

        with open(os.path.join("data", "pickled_countries.pickle"), "rb") as f:
            countries = pickle.load(f)

        with open(os.path.join("data", "pickled_locations.npy"), "rb") as f:
            numpy_arr = np.load(f)
        return users, countries, numpy_arr

    def save_numpy_matrix(self, useruuid_dict, country_dict, locations):
        with open("pickled_users.pickle", "wb") as f:
            pickle.dump(useruuid_dict, f)
        with open("pickled_countries.pickle", "wb") as f:
            pickle.dump(country_dict, f)
        with open("pickled_locations.npy", "wb") as f:
            np.save(f, locations)

    def generate_numpy_matrix_from_json(self):
        useruuid_dict = {}
        country_dict = {}

        user_count = 0
        country_count = 0
        rows = []
        locations = []

        def callback_func(row): rows.append(row)
        self.generate_location_data_from_json(callback_func)

        for row in tqdm(rows):
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

        self.save_numpy_matrix(useruuid_dict, country_dict, locations)

        return locations
