#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import FileLoader
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import DatabaseHelper
from dateutil import parser
import math
from collections import defaultdict
import pprint
import collections
d = DatabaseHelper.DatabaseHelper()
file_loader = FileLoader.FileLoader()
from dateutil.relativedelta import relativedelta
import numpy as np


def load_data():
    country_users = d.get_users_in_country("Japan")
    rows = []
    file_loader.generate_app_data_from_json(lambda r: rows.append(r))
    feature = "com.android.incallui"
    rows = [r for r in rows if r["package_name"] == feature and r["useruuid"] in country_users]
    return rows


def by_all_days(bin_size, times, rows):
    sns.set(color_codes=True)
    sns.set(style="white", palette="muted")
    start_end_pairs = [(parser.parse(r["start_time"]), parser.parse(r["end_time"])) for r in rows]
    values = []
    for start, end in start_end_pairs:
        for x in range(int((start.hour*60+start.minute)/bin_size), math.ceil((end.hour*60+end.minute)/bin_size)):
            values.append(x)

    ax = sns.distplot(values, bins=(24*60+60)/bin_size)
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels(times, rotation=90)
    ax.set(title="All days")
    ax.set(xlabel="time of day")
    sns.plt.tick_params(labelsize=14)

    [label.set_visible(False) for label in ax.xaxis.get_ticklabels()]

    for label in ax.xaxis.get_ticklabels()[::8]:
        label.set_visible(True)
    sns.plt.show()


def by_week_day(bin_size, times, rows):
    sns.set(color_codes=True)
    sns.set(style="white", palette="muted")
    for plot in range(7):
        plt.subplot(3, 3, plot+1)
        start_end_pairs = [(parser.parse(r["start_time"]), parser.parse(r["end_time"])) for r in rows if parser.parse(r["start_time"]).weekday() == plot]
        min_start = min([start for start, _ in start_end_pairs])
        min_start = min_start.replace(hour=0, minute=0, second=0)
        values = []
        for start, end in start_end_pairs:
            for x in range(int((start.hour*60+start.minute)/bin_size), math.ceil((end.hour*60+end.minute)/bin_size)):
                values.append(x)

        ax = sns.distplot(values, bins=(24*60+60)/bin_size)
        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels(times, rotation=90)
        ax.set(xlabel="clock")
        ax.set(title=start_end_pairs[0][0].strftime("%A"))
        sns.plt.tick_params(labelsize=14)

        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()]

        for label in ax.xaxis.get_ticklabels()[::8]:
            label.set_visible(True)
    sns.plt.show()


def get_times(bin_size):
    times = []
    dummy_date = parser.parse("2016-04-25 00:00:00")
    for t in range(math.ceil((60*24)/bin_size)):
        temp_time = dummy_date + relativedelta(minutes=(bin_size*t))
        times.append(temp_time.strftime("%H:%M"))
    return times

if __name__ == '__main__':
    bin_size = 5
    data = load_data()
    times = get_times(bin_size)
    #by_all_days(bin_size, times, data)
    by_week_day(bin_size, times, data)
