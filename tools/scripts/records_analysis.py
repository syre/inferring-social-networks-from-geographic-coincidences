#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import DatabaseHelper
import Predictor
import seaborn as sns
import numpy as np
import math
from dateutil.rrule import rrule, WEEKLY, DAILY
from dateutil import parser
from datetime import timedelta
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def get_data_for_heat_map(country="Japan", user=""):
    d = DatabaseHelper.DatabaseHelper()
    sept_min_datetime2 = "2015-08-30 00:00:00+00:00" #Mandag
    nov_max_datetime2 = "2015-12-13 23:59:59+00:00"
    number_of_month = 3
    days_in_week = 7
    number_of_weeks = 6
    data = np.full((number_of_month, days_in_week, number_of_weeks),
                   -1, dtype=int)
    sep_start = parser.parse(sept_min_datetime2)
    nov_end = parser.parse(nov_max_datetime2)
    weeks = list(rrule(freq=WEEKLY, dtstart=sep_start, until=nov_end))
    month_range = [9, 10, 11]
    month_index = 0
    for week_index, week in enumerate(tqdm(weeks)): #Vi starter på en mandag!!!
        current_month = week.month #mandag i ugen
        if current_month in month_range:
            month_index = month_range.index(current_month)
        break_flag = False
        days_in_week = list(rrule(freq=DAILY, dtstart=week, until=weeks[week_index+1]))
        for day_index, day in enumerate(days_in_week):
            if day.month not in month_range: #hvis dagens måned ikke er i listen
                if day.month > max(month_range):
                    break_flag = True
                    break #Break all!!!!
            else:
                #print("{}/{} - {}".format(day.day, day.month, day.weekday()))
                if day.month > current_month and day.month != 8:
                    current_month += 1
                    if current_month > max(month_range):
                        break_flag = True
                        break
                    else:
                        month_index = month_range.index(current_month)
                    start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                    end =  day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                    user_query = ""
                    if user != "":
                        user_query = " AND useruuid = '"+user+"'"
                    query = "select count(*) FROM location WHERE country='"+country+"' AND start_time >= '"+start+"' AND start_time <= '" + end + "'"+user_query+";"
                    result = d.run_specific_query(query)[0][0] 
                    data[month_index][day.weekday()][week_of_month(day)-1] = result
                    
                else: #Hvis vi er i current_month
                    if day.month != 8:
                        start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                        end =  day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                        user_query = ""
                        if user != "":
                            user_query = " AND useruuid = '"+user+"'"
                        query = "select count(*) FROM location WHERE country='"+country+"' AND start_time >= '"+start+"' AND start_time <= '" + end + "'" + user_query+";"
                        result = d.run_specific_query(query)[0][0] 
                        data[month_index][day.weekday()][week_of_month(day)-1] = result
        if break_flag:
            break 

    print(data)
    mask = data == -1
    print(mask)
    return data, mask

def week_of_month(date):
    flag = False
    if date.weekday() == 6:
        flag = True
    lst = list(rrule(freq=DAILY, dtstart=date-timedelta(days=(date.day-1)), until=date))
    d = defaultdict(int)
    for date in lst:
        d[date.weekday()]+=1
    for i in range(7):
        if i not in d:
            d[i] = 0
    if flag:
        return d[6]
    return d[6]+1


def calc_labels(data, bins): 
    max_value = max(data)
    min_value = min(data)
    step = math.ceil((max_value - min_value)/bins)
    xlabels = list(np.arange(min_value, max_value, step))

    #print("min_value = {}, max_value = {}, len = {}".format(min_value, max_value, len(labels)))
    return xlabels, step


def cdf_xy_plot(data, title, xlabel, ylabel, legend_labels):
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.plot(data[0][0], data[0][1], label=legend_labels[0])
    plt.plot(data[1][0], data[1][1], label=legend_labels[1], color='r')
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    plt.ylim(0, 1)
    xlim_max = max([max(data[0][0]), max(data[1][0])])
    plt.ylim(0, 1)
    plt.xlim(0, xlim_max)
    plt.title(title)
    plt.legend(prop={'size': 28})
    [item.set_fontsize(35) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(40)
    [item.set_fontsize(28) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()


def records_dist_plot(data, bins, xlabels, ylabels, titles, labels):
    #labels = calc_labels(data, bins)
    #print(labels)
    #fig, ax = sns.plt.subplots(2, 1)
    sns.set(font_scale=2.5)
    for index, sub_data in enumerate(data):
        sns.plt.subplot(2, 1, index+1)
        #print(labels[index])
        xticks, bin_size = calc_labels(sub_data, bins)
        print("title = {}\nbin_size = {}".format(titles[index], bin_size))
        ax = sns.distplot(sub_data, norm_hist=True, kde=False, bins=bins, hist_kws={'cumulative':True})
        #ax.set_xticklabels(calc_labels(sub_data, bins), rotation=90)
        sns.plt.xticks(xticks, rotation=90)

        #for label in [item.get_text() for item in ax.get_xticklabels()]:
        #    print("|{}|".format(label))
        print(list(np.arange(0, 1.1, 0.1)))
        sns.plt.yticks(list(np.arange(0, 1.1, 0.1)))
        #sns.plt.ylim((0.1))
        
        sns.plt.tick_params(labelsize=20)
        ax.set_xlabel(xlabels[index])
        ax.set_ylabel(ylabels[index])
        ax.set_title(titles[index])
    sns.plt.show()


def heat_map(data, mask, xlabels, ylabels, title="", anno=True):
    sns.set(font_scale=2.5)
    
    months = ["September", "October", "November"]
    max_val = np.amax(data)
    fig, ax = sns.plt.subplots(3, 1)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for index, ax in enumerate(ax.flat):
        sns.heatmap(data[index], ax=ax, xticklabels=[" "]*len(xlabels[index]) if index != 2 else xlabels[index],
                    annot=anno, fmt="d",
                    yticklabels=ylabels[index], mask=mask[index], vmin=0,
                    vmax=max_val, cbar=index == 0, cbar_ax=None if index else cbar_ax)
        if title != "" and index == 0:
            ax.set_title(title)
        ax.set_ylabel(months[index])
        #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    sns.plt.show()


def boxplot(data, xx, yy, title):
    sns.set(font_scale=2.5)
    ax = sns.boxplot(x=xx, y=yy, data=data)
    ax.set_title(title)
    sns.plt.show()


def data_summary_per_user(data, total_counts):
    q = data.groupby('country')['Location updates'].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    means = data.groupby('country')['Location updates'].mean()
    print("///////// Japan ////////")
    print(q['Japan']/total_counts['Japan'])
    print("////////")
    print("///////// Sweden //////////")
    print(q['Sweden']/total_counts['Sweden'])
    print("////////")
    print(means['Japan']/total_count['Japan'])
    print(means['Sweden']/total_count['Sweden'])

if __name__ == '__main__':

    countries = ["Japan", "Sweden"]
    d = DatabaseHelper.DatabaseHelper()

    #-------------- DIST PLOT --------------------#
    data = []
    titles = []
    labels = []
    xlabels = []
    dd = {'Location updates': [], 'country': []}
    total_count = {}
    for i, country in enumerate(countries):
        query = "SELECT useruuid, count(*) FROM location WHERE country='"+country+"' \
                 GROUP BY useruuid ORDER BY useruuid"
        result = d.run_specific_query(query)
        total_count[country] = len(result)
        print("country = {}, sum = {}, len = {}".format(country, sum([row[1] for row in result]), len(result)))
        for row in result: 
            data.append(row[1])
            dd['Location updates'].append(row[1])
            dd['country'].append(countries[i])
        titles.append("Cumulative distribution of location updates for users in " + country)
        labels.append(country)
        xlabels.append("Number of location updates\n"+"("+chr(ord('a') + i)+")")
    ylabels = ["Cumulative frequency of users", "Cumulative frequency of users"] #["Frequency (%)", "Frequency (%)"] #["Number of users", "Number of users"]
    df = pd.DataFrame(dd)
    data_summary_per_user(df, total_count)
    #boxplot(df, "country", "Location updates", "Location updates for all users in Japan and Sweden")
    #records_dist_plot(data, 100, xlabels, ylabels, titles, labels)
    #
    data_cdf = []
    for i, country in enumerate(countries):
        hist, bin_edges = np.histogram(data[i], bins=100, normed=True)
        cdf = np.around(np.cumsum(hist)*np.ceil(bin_edges[1]-bin_edges[0]), decimals=3)
        bin_edges = bin_edges[1:]
        data_cdf.append((bin_edges, cdf))

    #cdf_xy_plot(data_cdf, "CDF for location updates", "Location updates",
    #            "Frequency", countries)

    #............... HEAT-MAP --------------------#
    for country in countries:
        user = ""
        #data, mask = get_data_for_heat_map(country)
        title = "Number of location updates in "+country
        if user != "":
            title += " for user " + str(user)
        ylabels = [["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"]]
        xlabels = [["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]]
        #heat_map(data, mask, xlabels, ylabels, title, True)
