#!/usr/bin/env python3
"""
Finds users with locations for all days of the three months
"""
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DatabaseHelper import DatabaseHelper

db = DatabaseHelper()
plt.style.use("ggplot")

first_period_datetime_min = "2015-09-01 00:00:00+00:00"
first_period_time_bin_min = db.calculate_time_bins(first_period_datetime_min)[0]
first_period_datetime_max = "2015-09-30 23:59:59+00:00"
first_period_time_bin_max = db.calculate_time_bins(first_period_datetime_max)[0]
second_period_datetime_min = "2015-10-01 00:00:00+00:00"
second_period_time_bin_min = db.calculate_time_bins(second_period_datetime_min)[0]
second_period_datetime_max = "2015-10-31 23:59:59+00:00"
second_period_time_bin_max = db.calculate_time_bins(second_period_datetime_max)[0]
third_period_datetime_min = "2015-11-01 00:00:00+00:00"
third_period_time_bin_min = db.calculate_time_bins(third_period_datetime_min)[0]
third_period_datetime_max = "2015-11-30 23:59:59+00:00"
third_period_time_bin_max = db.calculate_time_bins(third_period_datetime_max)[0]


def generate_user_timebins_dict(country):
    locations = db.get_locations_by_country_only(country)
    users = set([l[0] for l in locations])
    user_timebins_dict = defaultdict(lambda: defaultdict(set))
    for l in locations:
        timebins = l[4]
        for timebin in timebins:
            if timebin > first_period_time_bin_min and timebin < first_period_time_bin_max:
                user_timebins_dict[l[0]]["first_period"].add(timebin)
            elif timebin > second_period_time_bin_min and timebin < second_period_time_bin_max:
                user_timebins_dict[l[0]]["second_period"].add(timebin)
            elif timebin > third_period_time_bin_min and timebin < third_period_time_bin_max:
                user_timebins_dict[l[0]]["third_period"].add(timebin)
    return user_timebins_dict


def user_timebins_to_percentages(user_timebins_dict):
    for user, periods_dict in user_timebins_dict.items():
        periods_dict["first_period"] = len(periods_dict["first_period"])/(first_period_time_bin_max-first_period_time_bin_min)
        periods_dict["second_period"] = len(periods_dict["second_period"])/(second_period_time_bin_max-second_period_time_bin_min)
        periods_dict["third_period"] = len(periods_dict["third_period"])/(third_period_time_bin_max-third_period_time_bin_min)
    return user_timebins_dict


def compute_plot_in_three_periods(country):
    user_timebins_dict = generate_user_timebins_dict(country)
    user_timebins_dict = user_timebins_to_percentages(user_timebins_dict, 0.4)

    def plot(x): return len([(user, p) for user, p in user_timebins_dict.items() if p["first_period"] > x and p["second_period"] > x and p["third_period"] > x])
    return [plot(x) for x in np.arange(0, 1, 0.01)]


def generate_plot():
    sweden, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Sweden"), label="Sweden - All months", color='#990033')
    japan, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Japan"), label="Japan - All months", color='#333399')

    first_period_datetime_min = "2015-09-01 00:00:00+00:00"
    first_period_time_bin_min = db.calculate_time_bins(first_period_datetime_min)[0]
    first_period_datetime_max = "2015-09-09 23:59:59+00:00"
    first_period_time_bin_max = db.calculate_time_bins(first_period_datetime_max)[0]
    second_period_datetime_min = "2015-09-10 00:00:00+00:00"
    second_period_time_bin_min = db.calculate_time_bins(second_period_datetime_min)[0]
    second_period_datetime_max = "2015-09-19 23:59:59+00:00"
    second_period_time_bin_max = db.calculate_time_bins(second_period_datetime_max)[0]
    third_period_datetime_min = "2015-09-20 00:00:00+00:00"
    third_period_time_bin_min = db.calculate_time_bins(third_period_datetime_min)[0]
    third_period_datetime_max = "2015-09-30 23:59:59+00:00"
    third_period_time_bin_max = db.calculate_time_bins(third_period_datetime_max)[0]

    sweden_sept, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Sweden"), label="Sweden - september", color='#990033', linestyle='--')
    japan_sept, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Japan"), label="Japan - september", color='#333399', linestyle='--')

    first_period_datetime_min = "2015-10-01 00:00:00+00:00"
    first_period_time_bin_min = db.calculate_time_bins(first_period_datetime_min)[0]
    first_period_datetime_max = "2015-10-09 23:59:59+00:00"
    first_period_time_bin_max = db.calculate_time_bins(first_period_datetime_max)[0]
    second_period_datetime_min = "2015-10-10 00:00:00+00:00"
    second_period_time_bin_min = db.calculate_time_bins(second_period_datetime_min)[0]
    second_period_datetime_max = "2015-10-19 23:59:59+00:00"
    second_period_time_bin_max = db.calculate_time_bins(second_period_datetime_max)[0]
    third_period_datetime_min = "2015-10-20 00:00:00+00:00"
    third_period_time_bin_min = db.calculate_time_bins(third_period_datetime_min)[0]
    third_period_datetime_max = "2015-10-31 23:59:59+00:00"
    third_period_time_bin_max = db.calculate_time_bins(third_period_datetime_max)[0]


    sweden_oct, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Sweden"), label="Sweden - october", color='#990033', linestyle="None", marker="x")
    japan_oct, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Japan"), label="Japan - october", color='#333399', linestyle="None", marker="x")

    first_period_datetime_min = "2015-11-01 00:00:00+00:00"
    first_period_time_bin_min = db.calculate_time_bins(first_period_datetime_min)[0]
    first_period_datetime_max = "2015-11-09 23:59:59+00:00"
    first_period_time_bin_max = db.calculate_time_bins(first_period_datetime_max)[0]
    second_period_datetime_min = "2015-11-10 00:00:00+00:00"
    second_period_time_bin_min = db.calculate_time_bins(second_period_datetime_min)[0]
    second_period_datetime_max = "2015-11-19 23:59:59+00:00"
    second_period_time_bin_max = db.calculate_time_bins(second_period_datetime_max)[0]
    third_period_datetime_min = "2015-11-20 00:00:00+00:00"
    third_period_time_bin_min = db.calculate_time_bins(third_period_datetime_min)[0]
    third_period_datetime_max = "2015-11-30 23:59:59+00:00"
    third_period_time_bin_max = db.calculate_time_bins(third_period_datetime_max)[0]


    sweden_nov, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Sweden"), label="Sweden - november", color='#990033', linestyle="None", marker=".")
    japan_nov, = plt.plot(np.arange(0, 1, 0.01), compute_plot_in_three_periods("Japan"), label="Japan - november", color='#333399', linestyle="None", marker=".")


    plt.legend(handles=[sweden, japan, sweden_sept, japan_sept, sweden_oct, japan_oct, sweden_nov, japan_nov], prop={'size': 20})
    plt.ylabel("Number of users")
    plt.xlabel("Percentage of total timebins")
    plt.title("Plot of users with unique timebins in all three months")
    plt.tick_params(labelsize=15)
    [item.set_fontsize(30) for item in [plt.gca().yaxis.label, plt.gca().xaxis.label]]
    plt.gca().title.set_fontsize(48)
    [item.set_fontsize(25) for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()]
    plt.show()

    user_locations_dict = {}


def get_users_with_unique_days(number):
    """
    Find test and train users with at least 'number' unique days in
    both test or train months
    """
    train_users = []
    test_users = []
    for u, location_dates in user_locations_dict.items():
        sept_locations = [x for x in location_dates if x[1] == 9]
        oct_locations = [x for x in location_dates if x[1] == 10]
        nov_locations = [x for x in location_dates if x[1] == 11]
        if len(set(sept_locations)) >= number and len(set(oct_locations)) >= number:
            train_users.append(u)
        if len(set(oct_locations)) >= number and len(set(nov_locations)) >= number:
            test_users.append(u)
    return train_users, test_users


def generate_plot(min=0, max=25):
    train_values = []
    test_values = []
    for x in range(min, max):
        train_users, test_users = get_users_with_unique_days(x)
        train_values.append(len(train_users))
        test_values.append(len(test_users))
    plt.title("Number of users with unique days with location updates (Sweden)")
    plt.xlabel("number of unique days with updates in both months")
    plt.ylabel("number of users")
    plt.plot(range(min, max), train_values, label="train users (sept, oct)")
    plt.plot(range(min, max), test_values, label="test users (oct, nov)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    first_period_datetime_min = "2015-09-01 00:00:00+00:00"
    first_period_time_bin_min = db.calculate_time_bins(first_period_datetime_min)[0]
    first_period_datetime_max = "2015-09-09 23:59:59+00:00"
    first_period_time_bin_max = db.calculate_time_bins(first_period_datetime_max)[0]
    second_period_datetime_min = "2015-09-10 00:00:00+00:00"
    second_period_time_bin_min = db.calculate_time_bins(second_period_datetime_min)[0]
    second_period_datetime_max = "2015-09-19 23:59:59+00:00"
    second_period_time_bin_max = db.calculate_time_bins(second_period_datetime_max)[0]
    third_period_datetime_min = "2015-09-20 00:00:00+00:00"
    third_period_time_bin_min = db.calculate_time_bins(third_period_datetime_min)[0]
    third_period_datetime_max = "2015-09-30 23:59:59+00:00"
    third_period_time_bin_max = db.calculate_time_bins(third_period_datetime_max)[0]
    user_timebins_dict = generate_user_timebins_dict("Sweden")
    user_timebins_dict = user_timebins_to_percentages(user_timebins_dict)
    prob = 0.4
    filtered = [(user, p) for user, p in user_timebins_dict.items() if p["first_period"] > prob and p["second_period"] > prob and p["third_period"] > prob]
    print([x[0] for x in filtered])
    print(len(filtered))