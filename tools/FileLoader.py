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

    def __init__(self):
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

    def generate_data_from_json(self, filenames, callback_func, path="data"):
        for file_name in tqdm(filenames):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data):
                callback_func(row)

    def generate_demographics_from_csv(self, path="data"):
        file_name = "user_data.csv"
        user_info_dict = defaultdict(list)

        with open(os.path.join(self.DATA_PATH, file_name)) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                birthdate = parser.parse(row[2])
                age = datetime.now()-birthdate
                user_info_dict[row[0]] = {"gender": row[1],
                                          "age": int(age.days/365),
                                          "birthdate": row[2]}
        return user_info_dict

    def filter_demographic_outliers(self, user_info_dict):
        user_info_dict = defaultdict(list, {k: v for k, v in
                                            user_info_dict.items()
                                            if v["birthdate"] !=
                                            "1990-01-01" and
                                            v["birthdate"] != "1981-01-01" and
                                            v["birthdate"] != "1980-12-01" and
                                            v["age"] > 16 and v["age"] < 100})
        return user_info_dict

    def generate_app_data_from_json(self, callback_func, path="data"):
        filenames = [
            "all_app_201509.json", "all_app_201510.json",
            "all_app_201511.json"]
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
        with open(os.path.join(self.DATA_PATH, "friend_pairs.pickle"),
                  "rb") as fp:
            friend_pairs = pickle.load(fp)
        with open(os.path.join(self.DATA_PATH, "nonfriend_pairs.pickle"),
                  "rb") as fp:
            nonfriend_pairs = pickle.load(fp)

        return friend_pairs, nonfriend_pairs

    def load_cooccurrences(self):
        with open(os.path.join(self.DATA_PATH, "cooccurrences.npy"),
                  "rb") as f:
            coocs = np.load(f)
        return coocs

    def save_cooccurrences(self, coocs):
        with open(os.path.join(self.DATA_PATH, "cooccurrences.npy"), "wb") as f:
            np.save(f, coocs)

    def save_x_and_y(self, x, y):
        with open(os.path.join(self.DATA_PATH, "datasetX.pickle"), "wb") as fp:
            pickle.dump(x, fp)
        with open(os.path.join(self.DATA_PATH, "data", "datasetY.pickle"),
                  "wb") as fp:
            pickle.dump(y, fp)

    def load_x_and_y(self):
        with open(os.path.join(self.DATA_PATH, "datasetX.pickle"), "rb") as fp:
            x = pickle.load(fp)
        with open(os.path.join(self.DATA_PATH, "datasetY.pickle"), "rb") as fp:
            y = pickle.load(fp)
        return x, y

    def load_numpy_matrix(self):
        with open(os.path.join(self.DATA_PATH, "pickled_users.pickle"),
                  "rb") as f:
            users = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_countries.pickle"),
                  "rb") as f:
            countries = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_locations.npy"),
                  "rb") as f:
            numpy_arr = np.load(f)
        return users, countries, numpy_arr

    def save_numpy_matrix(self, useruuid_dict, country_dict, locations):
        with open(os.path.join(self.DATA_PATH, "pickled_users.pickle"),
                  "wb") as f:
            pickle.dump(useruuid_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_countries.pickle"),
                  "wb") as f:
            pickle.dump(country_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_locations.npy"),
                  "wb") as f:
            np.save(f, locations)

    def load_missing_data(self):
        with open(os.path.join("data", "missing_data.json"), 'r') as json_file:
            return json.load(json_file)

    def load_by_filename(self, filename):
        dot = filename.rfind(".")
        if filename[dot+1:] == "json":
            with open(os.path.join("data", filename), 'r') as json_file:
                return json.load(json_file)
        elif filename[dot+1:] == "pickle":
            with open(os.path.join("data", filename), "rb") as f:
                return pickle.load(f)
        else:
            with open(os.path.join("data", filename), "r") as f:
                return f.readlines()
