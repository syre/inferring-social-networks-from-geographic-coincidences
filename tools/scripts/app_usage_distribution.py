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


def generate_subplot_numbers(total_plot):
    row = 1
    col = 1
    flag = True
    if math.sqrt(total_plot) % 1 == 0:
        return {'rows': int(math.sqrt(total_plot)),
                'cols': int(math.sqrt(total_plot))}

    while row*col<total_plot:
        if flag:
            if col % 2 == 0:
                col += 1
            else:
                if col > 1 and not col % 2 == 1:
                    col -= 1
                row += 1
                flag = not flag
        else:
            if row % 2 == 0:
                row += 1
            else:
                if row > 1 and not row % 2 == 1:
                    row -= 1
                col += 1
                flag = not flag
        print("Total plots: {}\nrow: {}\ncol: {}\n-------------".format(total_plot, row, col))
    return {'rows': row, 'cols': col}


def load_data(features):
    country_users = d.get_users_in_country("Japan")
    rows = []
    file_loader.generate_app_data_from_json(lambda r: rows.append(r))
    result = defaultdict(list)

    for r in rows:
        if r["package_name"] in features and r["useruuid"] in country_users:
            result[r["package_name"]].append(r)
    
    #result = [result[r["package_name"]].append(r) for r in rows
    #          if r["package_name"] in features and r["useruuid"]
    #          in country_users]
    #rows = [r for r in rows if r["package_name"]
    #        in features and r["useruuid"] in country_users]
    return result


def by_all_days(bin_size, times, rows_dict):
    sns.set(color_codes=True)
    sns.set(style="white", palette="muted")
    sns.set(font_scale=1.5)
    plot_info = generate_subplot_numbers(len(rows_dict.keys()))
    plot = 0
    for feature, rows in rows_dict.items():
        plt.subplot(plot_info['rows'], plot_info['cols'], plot+1)
        start_end_pairs = [
            (parser.parse(r["start_time"]), parser.parse(r["end_time"])) for r in rows]
        values = []
        for start, end in start_end_pairs:
            for x in range(int((start.hour*60+start.minute)/bin_size), math.ceil((end.hour*60+end.minute)/bin_size)):
                values.append(x)
        #print("start_end_pairs: {}".format(start_end_pairs))        
        print("len(values): {}".format(len(values)))
        print(values[:10])
        print("feature: "+feature)
        #if len(values) > 1:
        #    print("if!!!")
        #    values.append(-1)
        #try:

        if len(values) > 1:
            ax = sns.distplot(values, bins=(24*60+60)/bin_size, kde=False, norm_hist=True)
            ax.set_xticks(np.arange(len(times)))
            ax.set_xticklabels(times, rotation=90)
            ax.set(title="All days ["+feature+"]")
            ax.set(xlabel="time of day")
            ax.set(ylabel="Probability")
            
            sns.plt.tick_params(labelsize=14)

            [label.set_visible(False) for label in ax.xaxis.get_ticklabels()]

            for label in ax.xaxis.get_ticklabels()[::8]:
                label.set_visible(True)
        #except Exception as e:
        #    print(str(e))
        plot += 1
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    sns.plt.show()


def by_week_day(bin_size, times, rows):
    sns.set(color_codes=True)
    sns.set(style="white", palette="muted")
    for plot in range(7):
        plt.subplot(3, 3, plot+1)
        start_end_pairs = [(parser.parse(r["start_time"]), parser.parse(
            r["end_time"])) for r in rows if parser.parse(r["start_time"]).weekday() == plot]
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
    im_features = ['com.snapchat.android', 'com.Slack',
                   'com.verizon.messaging.vzmsgs', 'jp.naver.line.android',
                   'com.whatsapp', 'org.telegram.messenger',
                   'com.google.android.talk', 'com.viber.voip',
                   'com.alibaba.android.rimet',
                   'com.skype.raider', 'com.sonyericsson.conversations',
                   'com.kakao.talk', 'com.google.android.apps.messaging',
                   'com.facebook.orca', 'com.tenthbit.juliet',
                   'com.tencent.mm']
    data = load_data(im_features)
    plot_info = generate_subplot_numbers(len(im_features))
    print(len(im_features))
    print(len(data.keys()))
    print(set([feature for feature in data.keys()]).symmetric_difference(set(im_features)))
    #pprint.pprint(plot_info)
    times = get_times(bin_size)
    by_all_days(bin_size, times, data)
    #by_week_day(bin_size, times, data)
    