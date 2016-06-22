#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DatabaseHelper import DatabaseHelper
from DatasetHelper import DatasetHelper
import csv

database_helper = DatabaseHelper()
dataset_helper = DatasetHelper()

COUNTRY = "Sweden"

filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                      [(17.9529121, 59.4050982), 1000],
                                      [(11.9387722, 57.7058472), 1000]],
                           "Japan": [[(139.743862, 35.630338), 1000]]}

first_period_datetime_min = "2015-09-01 00:00:00+02:00"
first_period_time_bin_min = database_helper.calculate_time_bins(first_period_datetime_min)[0]

last_period_datetime_max = "2015-11-30 23:59:59+02:00"
last_period_time_bin_max = database_helper.calculate_time_bins(last_period_datetime_max)[0]
# retrieve locations
#users, countries, locations = database_helper.generate_numpy_matrix_from_database()

users, countries, locations = database_helper.generate_numpy_matrix_from_database(filter_places_dict[COUNTRY])
# filter by country
locations = locations[locations[:, 3] == countries[COUNTRY]]
coocs = dataset_helper.generate_cooccurrences_array(locations)

coocs = coocs[coocs[:, 3] > first_period_time_bin_min]
coocs = coocs[coocs[:, 3] <= last_period_time_bin_max]

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "{}_graph.csv".format(COUNTRY).lower()), "w+", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        for row in coocs:
                writer.writerow([row[0], row[1]])
