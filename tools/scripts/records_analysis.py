#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import DatabaseHelper
import Predictor
import pprint
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import math
from dateutil.rrule import rrule, WEEKLY, DAILY
from dateutil import parser
from datetime import timedelta
import calendar
from collections import defaultdict
from tqdm import tqdm

def get_data_for_heat_map():
    d = DatabaseHelper.DatabaseHelper()
    p = Predictor.Predictor()
    sept_min_datetime = "2015-09-01 00:00:00+00:00"
    sept_min_datetime2 = "2015-08-30 00:00:00+00:00" #Mandag
    sept_min_time_bin = d.calculate_time_bins(sept_min_datetime, sept_min_datetime)[0]
    sept_max_datetime = "2015-09-30 23:59:59+00:00"
    sept_max_time_bin = d.calculate_time_bins(sept_max_datetime, sept_max_datetime)[0]
    oct_min_datetime = "2015-10-01 00:00:00+00:00"
    oct_min_time_bin = d.calculate_time_bins(oct_min_datetime, oct_min_datetime)[0]
    oct_max_datetime = "2015-10-31 23:59:59+00:00"
    oct_max_time_bin = d.calculate_time_bins(oct_max_datetime, oct_max_datetime)[0]
    nov_min_datetime = "2015-11-01 00:00:00+00:00"
    nov_min_time_bin = d.calculate_time_bins(nov_min_datetime, nov_min_datetime)[0]
    nov_max_datetime = "2015-11-30 23:59:59+00:00"
    nov_max_datetime2 = "2015-12-13 23:59:59+00:00"
    nov_max_time_bin = d.calculate_time_bins(nov_max_datetime, nov_max_datetime)[0]
    

    print("processing users, countries and locations as numpy matrix (train)")
    users_train, countries_train, locations_train = d.generate_numpy_matrix_from_database()
    locations = p.filter_by_country(locations_train, countries_train)
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
                    time_bins = d.calculate_time_bins(day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00", day.strftime("%Y-%m-%d")+" 23:59:59+00:00")
                    result = locations[locations[:, 2] >= time_bins[0]]
                    result = result[result[:, 2] <= time_bins[-1]]
                    data[month_index][day.weekday()][week_of_month(day)-1] = result.shape[0]
                    
                else: #Hvis vi er i current_month
                    if day.month != 8:
                        time_bins = d.calculate_time_bins(day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00", day.strftime("%Y-%m-%d")+" 23:59:59+00:00")
                        result = locations[locations[:, 2] >= time_bins[0]]
                        result = result[result[:, 2] <= time_bins[-1]]
                        data[month_index][day.weekday()][week_of_month(day)-1] = result.shape[0]
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
    return list(np.arange(min_value, max_value, step))


def records_dist_plot(data, bins):
    labels = calc_labels(data, bins)
    print(labels)
    ax = sns.distplot(data, kde=False, norm_hist=True, bins=bins)
    #print(len(labels))
    #ax.set_xticks(np.arange(len(labels)))
    #ax.set_xticklabels(labels)
    sns.plt.tick_params(labelsize=20)
    sns.plt.show()


def heat_map(data, mask, xlabels, ylabels):
    sns.set(font_scale=2.5)
    for index, month_data in enumerate(data):
        sns.plt.subplot(3, 1, index+1)
        
        sns.heatmap(month_data, xticklabels=xlabels[index],
                    yticklabels=ylabels[index], mask=mask[index]) #mask=mask[index]
    sns.plt.show()




if __name__ == '__main__':
    d = DatabaseHelper.DatabaseHelper()
    query = "SELECT useruuid, count(*) FROM location WHERE country='Japan' \
             GROUP BY useruuid ORDER BY useruuid"
    #result = d.run_specific_query(query)
    #print(result[0])
    
    #counts = [row[1] for row in result]
    
    #print(counts[0])
    #records_dist_plot(counts, 200)
    #
    data, mask = get_data_for_heat_map()    
    ylabels = [["M", "T", "W", "T", "F", "S", "S"],
               ["M", "T", "W", "T", "F", "S", "S"],
               ["M", "T", "W", "T", "F", "S", "S"]]
    xlabels = [["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
               ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
               ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]]
    heat_map(data, mask, xlabels, ylabels)
