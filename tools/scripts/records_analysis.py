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

def get_data_for_heat_map(user=""):
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
                    query = "select count(*) FROM location WHERE start_time >= '"+start+"' AND start_time <= '" + end + "'"+user_query+";"
                    result = d.run_specific_query(query)[0][0] 
                    print(result)
                    data[month_index][day.weekday()][week_of_month(day)-1] = result
                    
                else: #Hvis vi er i current_month
                    if day.month != 8:
                        start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                        end =  day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                        user_query = ""
                        if user != "":
                            user_query = " AND useruuid = '"+user+"'"
                        query = "select count(*) FROM location WHERE start_time >= '"+start+"' AND start_time <= '" + end + "'" + user_query+";"
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
    return list(np.arange(min_value, max_value, step))


def records_dist_plot(data, bins):
    labels = calc_labels(data, bins)
    print(labels)
    ax = sns.distplot(data, kde=False, norm_hist=True, bins=bins)
    sns.plt.tick_params(labelsize=20)
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
    sns.plt.show()

if __name__ == '__main__':
    #-------------- DIST PLOT --------------------#
    d = DatabaseHelper.DatabaseHelper()
    query = "SELECT useruuid, count(*) FROM location WHERE country='Japan' \
             GROUP BY useruuid ORDER BY useruuid"
    #result = d.run_specific_query(query)
    #counts = [row[1] for row in result]
    #records_dist_plot(counts, 200)
    
    
    #............... HEAT-MAP --------------------#
    data, mask = get_data_for_heat_map()   
    title = "Number of location updates in Japan for user " #+ str(user)
    ylabels = [["M", "T", "W", "T", "F", "S", "S"],
               ["M", "T", "W", "T", "F", "S", "S"],
               ["M", "T", "W", "T", "F", "S", "S"]]
    xlabels = [["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
               ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
               ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]]
    heat_map(data, mask, xlabels, ylabels, title, True)
