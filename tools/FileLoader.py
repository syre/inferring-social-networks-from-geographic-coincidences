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
from dateutil import parser

class FileLoader():

    def __init__(self):
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

    def generate_data_from_json(self, filenames, callback_func, path=""):
        for file_name in tqdm(filenames):
            with open(os.path.join(self.DATA_PATH, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data):
                callback_func(row)

    def generate_demographics_from_csv(self, path="data"):
        file_name = "user_data.csv"
        user_info_dict = defaultdict(dict)

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
        user_info_dict = defaultdict(dict, {k: v for k, v in
                                            user_info_dict.items()
                                            if v["birthdate"] !=
                                            "1990-01-01" and
                                            v["birthdate"] != "1981-01-01" and
                                            v["birthdate"] != "1980-12-01" and
                                            v["age"] > 16 and v["age"] < 100})
        return user_info_dict

    def generate_app_data_from_json(self,
                                    callback_func,
                                    filenames=["all_app_201509.json",
                                               "all_app_201510.json",
                                               "all_app_201511.json"]):
        return self.generate_data_from_json(filenames, callback_func)

    def generate_location_data_from_json(self, callback_func):
        filenames = [
            "all_201509.json", "all_201510.json", "all_201511.json"]
        return self.generate_data_from_json(filenames, callback_func)

    def save_met_in_next_train(self, met_in_next_train):
        with open(os.path.join(self.DATA_PATH, "met_in_next_train.pickle"), "rb") as fp:
            pickle.dump(met_in_next_train, fp)

    def save_met_in_next_test(self, met_in_next_test):
        with open(os.path.join(self.DATA_PATH, "met_in_next_test.pickle"), "rb") as fp:
            pickle.dump(met_in_next_test, fp)
    
    def load_met_in_next_train(self):
        with open(os.path.join(self.DATA_PATH, "met_in_next_train.pickle"), "rb") as fp:
            met_in_next_train = pickle.load(fp)
        return met_in_next_train
    
    def load_met_in_next_test(self):
        with open(os.path.join(self.DATA_PATH, "met_in_next_test.pickle"), "rb") as fp:
            met_in_next_test = pickle.load(fp)
        return met_in_next_test


    def load_cooccurrences_train(self):
        with open(os.path.join(self.DATA_PATH, "cooccurrences_train.npy"),
                  "rb") as f:
            coocs = np.load(f)
        return coocs

    def load_cooccurrences_test(self):
        with open(os.path.join(self.DATA_PATH, "cooccurrences_test.npy"),
                  "rb") as f:
            coocs = np.load(f)
        return coocs

    def save_cooccurrences_train(self, coocs):
        with open(os.path.join(self.DATA_PATH, "cooccurrences_train.npy"), "wb") as f:
            np.save(f, coocs)

    def save_cooccurrences_test(self, coocs):
        with open(os.path.join(self.DATA_PATH, "cooccurrences_test.npy"), "wb") as f:
            np.save(f, coocs)

    def save_x_and_y(self, **kwargs):
        for key in kwargs: 
            with open(os.path.join(self.DATA_PATH, str(key)+".pickle"), "wb") as fp:
                pickle.dump(kwargs[key], fp)

    def load_x_and_y(self):
        with open(os.path.join(self.DATA_PATH, "X_train.pickle"), "rb") as fp:
            X_train = pickle.load(fp)
        with open(os.path.join(self.DATA_PATH, "y_train.pickle"), "rb") as fp:
            y_train = pickle.load(fp)
        with open(os.path.join(self.DATA_PATH, "X_test.pickle"), "rb") as fp:
            X_test = pickle.load(fp)
        with open(os.path.join(self.DATA_PATH, "y_test.pickle"), "rb") as fp:
            y_test = pickle.load(fp)
        return X_train, y_train, X_test, y_test

    def load_numpy_matrix_train(self):
        with open(os.path.join(self.DATA_PATH, "pickled_users_train.pickle"),
                  "rb") as f:
            users = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_countries_train.pickle"),
                  "rb") as f:
            countries = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_locations_train.npy"),
                  "rb") as f:
            numpy_arr = np.load(f)
        return users, countries, numpy_arr

    def load_numpy_matrix_test(self):
        with open(os.path.join(self.DATA_PATH, "pickled_users_test.pickle"),
                  "rb") as f:
            users = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_countries_test.pickle"),
                  "rb") as f:
            countries = pickle.load(f)

        with open(os.path.join(self.DATA_PATH, "pickled_locations_test.npy"),
                  "rb") as f:
            numpy_arr = np.load(f)
        return users, countries, numpy_arr

    def save_numpy_matrix_test(self, useruuid_dict, country_dict, locations):
        with open(os.path.join(self.DATA_PATH, "pickled_users_test.pickle"),
                  "wb") as f:
            pickle.dump(useruuid_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_countries_test.pickle"),
                  "wb") as f:
            pickle.dump(country_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_locations_test.npy"),
                  "wb") as f:
            np.save(f, locations)

    def save_numpy_matrix_train(self, useruuid_dict, country_dict, locations):
        with open(os.path.join(self.DATA_PATH, "pickled_users_train.pickle"),
                  "wb") as f:
            pickle.dump(useruuid_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_countries_train.pickle"),
                  "wb") as f:
            pickle.dump(country_dict, f)
        with open(os.path.join(self.DATA_PATH, "pickled_locations_train.npy"),
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

if __name__ == '__main__':
    fl = FileLoader()
    rows = []
    callback = lambda r: rows.append(r)
    fl.generate_data_from_json(["all_201509.json"], callback)
    
    print(min(rows, key=lambda r:parser.parse(r["start_time"]))["start_time"], max(rows,key=lambda r:parser.parse(r["end_time"]))["end_time"])

    rows = []
    callback = lambda r: rows.append(r)
    fl.generate_data_from_json(["all_201510.json"], callback)
    print(min(rows, key=lambda r:parser.parse(r["start_time"]))["start_time"], max(rows,key=lambda r:parser.parse(r["end_time"]))["end_time"])

    rows = []
    callback = lambda r: rows.append(r)
    fl.generate_data_from_json(["all_201511.json"], callback)
    print(min(rows, key=lambda r:parser.parse(r["start_time"]))["start_time"], max(rows,key=lambda r:parser.parse(r["end_time"]))["end_time"])
