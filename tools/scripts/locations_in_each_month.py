#!/usr/bin/env python3
"""
Finds users and the number of location updates for each month
as (user, num_of_missing_months, (locations_for_september, locations_for_october, locations_for_november))
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor

database_helper = DatabaseHelper()
predictor = Predictor("Japan")

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

users, countries, locations = database_helper.generate_numpy_matrix_from_database()
num_to_usr_dict = {value: key for (key, value) in users.items()}
japan_arr = predictor.filter_by_country(locations, countries)

sept_arr = japan_arr[japan_arr[:, 2] <= sept_max_time_bin]
sept_arr = sept_arr[sept_arr[:, 2] > sept_min_time_bin]

oct_arr = japan_arr[japan_arr[:, 2] <= oct_max_time_bin]
oct_arr = oct_arr[oct_arr[:, 2] > oct_min_time_bin]

nov_arr = japan_arr[japan_arr[:, 2] <= nov_max_time_bin]
nov_arr = nov_arr[nov_arr[:, 2] > nov_min_time_bin]

print(japan_arr.shape[0])

print(sept_arr.shape[0])
print(oct_arr.shape[0])
print(nov_arr.shape[0])

users = []
japan_users = set(list(japan_arr[:,0].reshape(1, -1)[0]))
print(len(japan_users))
for user in japan_users:
    count = 0
    sept_val = sept_arr[sept_arr[:, 0] == user].shape[0]
    oct_val = oct_arr[oct_arr[:, 0] == user].shape[0]
    nov_val = nov_arr[nov_arr[:, 0] == user].shape[0]
    if sept_arr[sept_arr[:, 0] == user].shape[0] == 0:
        print("user with ids: {} and {}, not present in sept_arr".format(user, num_to_usr_dict[user]))
        count += 1
    if oct_arr[oct_arr[:, 0] == user].shape[0] == 0:
        print("user with ids: {} and {}, not present in oct_arr".format(user, num_to_usr_dict[user]))
        count += 1
    if nov_arr[nov_arr[:, 0] == user].shape[0] == 0:
        print("user with ids: {} and {}, not present in nov_arr".format(user, num_to_usr_dict[user]))
        count += 1
    users.append((user,count, (sept_val, oct_val, nov_val)))
print(len([u for u in users if u[1] == 0]))
print(len([u for u in users if u[1] == 1]))
print(len([u for u in users if u[1] == 2]))
print(len([u for u in users if u[1] == 3]))

print([(num_to_usr_dict[u[0]], u[2]) for u in users if u[1] == 0])