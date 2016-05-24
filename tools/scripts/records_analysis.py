#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import DatabaseHelper
import DatasetHelper
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
import matplotlib.ticker as tck
import pandas as pd
import collections
import pprint
import itertools
from tqdm import tqdm


def get_data_for_heat_map_per_month(country="Japan", total_count={}, user=""):
    """
        Generate numpy matrix represent heatmap with the following dimension: 3x7x6 (3 month, 7 days in a week, up to 6 weeks in a month)
        Hardcoded datetimes

        Arguments:
            country {string} -- Which country to get data from
            total_count {dict} -- Dict with total users in country (key = country, values = total users)
            user {string} -- find it for specific user

        Returns:
            numpy array -- Numpy array with number of location in that day, in that week, in that month
    """
    d = DatabaseHelper.DatabaseHelper()
    timezones = {"Sweden": "+02:00",
                 "Japan": "+09:00"}
    sept_min_datetime2 = "2015-08-30 00:00:00" + timezones[country] #Mandag
    nov_max_datetime2 = "2015-12-13 23:59:59" + timezones[country]
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
                    start = day.strftime("%Y-%m-%d %H:%M:%S") + timezones[country]
                    end =  day.strftime("%Y-%m-%d")+" 23:59:59" + timezones[country]
                    user_query = ""
                    if user != "":
                        user_query = " AND useruuid = '"+user+"'"
                    query = "select count(*) FROM location WHERE country='" + country + \
                            "' AND ((start_time >= '" + start + \
                            "' AND start_time < '" + end + "')" + \
                            " OR (end_time >= '" + start + \
                            "' AND end_time < '" + start + "')" + \
                            " OR (start_time < '"+start + \
                            "' AND end_time > '"+end+"'))" + user_query+";"
                    result = d.run_specific_query(query)[0][0] 
                    data[month_index][day.weekday()][week_of_month(day)-1] = result/total_count[country]
                    
                else: #Hvis vi er i current_month
                    if day.month != 8:
                        start = day.strftime("%Y-%m-%d %H:%M:%S") + timezones[country]
                        end =  day.strftime("%Y-%m-%d")+" 23:59:59" + timezones[country]
                        user_query = ""
                        if user != "":
                            user_query = " AND useruuid = '"+user+"'"
                        query = "select count(*) FROM "+ \
                            "location WHERE country='" + country + \
                            "' AND ((start_time >= '" + start + \
                            "' AND start_time < '" + end + "')" + \
                            " OR (end_time >= '" + start + \
                            "' AND end_time < '" + start + "')" + \
                            " OR (start_time < '"+start + \
                            "' AND end_time > '"+end+"'))" + user_query+";"
                        result = d.run_specific_query(query)[0][0] 
                        data[month_index][day.weekday()][week_of_month(day)-1] = result/total_count[country]
        if break_flag:
            break 

    print(data)
    mask = data == -1
    print(mask)
    return data, mask


def get_all_users_in_country(country, users=[]):
    """
        Generate dict with incremental values as key, and original useruuid as values AND vice versa

        Arguments:
            country {string} -- Which country to get users from
            users {list} -- List of users. If not empty list, the db call is omitted

        Returns:
            dict -- dict with incremental values as key, and original useruuid as values AND vice versa
    """
    d = DatabaseHelper.DatabaseHelper()
    if not users:
        result = d.run_specific_query("SELECT DISTINCT(useruuid) FROM location WHERE country='"+country+"' ORDER BY useruuid")
        users = [row[0] for row in result]
    alt_useruid = {}
    i = 0
    for user in users:
        if user not in alt_useruid:
            alt_useruid[user] = i
            alt_useruid[i] = user
            i += 1

    return alt_useruid


def get_data_for_heat_map_per_user(country, start_date, end_date, input_users=[], counts=False):
    """
        Generate numpy matrix represent heatmap with the following dimension: usersxdays

        Arguments:
            country {string} -- Which country to get data from
            start_date {string} -- start datetime
            end_time {string} -- end datetime
            input_users {list} -- list of specific users
            counts {boolean} -- Indicates if it should find number of locations (True) or just if the user has an update (False)

        Returns:
            numpy array -- Numpy array
    """
    d = DatabaseHelper.DatabaseHelper()
    users = get_all_users_in_country(country, input_users)
    sept_min_datetime2 = start_date + " 00:00:00+00:00" #Mandag
    nov_max_datetime2 = end_date + " 23:59:59+00:00"
    sep_start = parser.parse(sept_min_datetime2)
    nov_end = parser.parse(nov_max_datetime2)
    days = list(rrule(freq=DAILY, dtstart=sep_start, until=nov_end))
    #print("get_data_for_heat_map_per_user")
    #print(len(users)/2)
    data = np.full((len(users)/2, len(days)),
                   0, dtype=int)
    timezones = {"Sweden": "+02:00",
                 "Japan": "+09:00"}
    if not input_users:
        for day_index, day in enumerate(tqdm(days)): #Vi starter på en mandag!!!
            start = day.strftime("%Y-%m-%d %H:%M:%S") + timezone[country]
            end =  day.strftime("%Y-%m-%d")+" 23:59:59" + timezones[country]
            result = d.run_specific_query("SELECT useruuid, count(*) FROM " +
                                          "location WHERE country='" + country +
                                          "' AND ((start_time >= '" + start +
                                          "' AND start_time < '" + end + "')" +
                                          " OR (end_time >= '" + start +
                                          "' AND end_time < '" + start + "')" +
                                          " OR (start_time < '"+start +
                                          "' AND end_time > '"+end+"'))" +
                                          "GROUP BY useruuid")
            for row in result:
                user = row[0]
                if counts:
                    data[users[user]][day_index] = row[1]
                else:
                    data[users[user]][day_index] = 1
    else:
        for day_index, day in enumerate(tqdm(days)): #Vi starter på en mandag!!!
            for user in input_users:
                start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                end = day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                result = d.run_specific_query("SELECT count(*) FROM " +
                                              "location WHERE country='" + country +
                                              "' AND ((start_time >= '" + start +
                                              "' AND start_time < '" + end + "')" +
                                              " OR (end_time >= '" + start +
                                              "' AND end_time < '" + start + "')" +
                                              " OR (start_time < '"+start +
                                              "' AND end_time > '"+end+"'))" +
                                              " AND useruuid='"+user+"'")
                for row in result:
                    if row[0] > 0:
                        if counts:
                            data[users[user]][day_index] = row[0]
                        else:
                            data[users[user]][day_index] = 1
                        
    return data, users


def get_data_for_heat_map_per_user_aggregated(country, start_date, end_date, bins=60, input_users=[], counts=False):
    """
        Generate numpy matrix represent heatmap, with user and number of bins over 24 hours.
        Bins is number of minutes each bin is. So dimension for arrays is usersx((24*60)/bins)

        Arguments:
            country {string} -- Which country to get data from
            start_date {string} -- start datetime
            end_time {string} -- end datetime
            input_users {list} -- list of specific users
            counts {boolean} -- Indicates if it should find number of locations (True) or just if the user has an update (False)

        Returns:
            numpy array -- Numpy array
    """
    d = DatabaseHelper.DatabaseHelper()
    users = get_all_users_in_country(country, input_users)
    start_time = parser.parse(start_date + " 00:00:00+00:00")
    end_time = parser.parse(end_date + " 23:59:59+00:00")
    days = list(rrule(freq=DAILY, dtstart=start_time, until=end_time))
    data = np.full((len(users)/2, (24*60)/bins),
                   0, dtype=int)
    timezones = {"Sweden": "+02:00",
                 "Japan": "+09:00"}
    if not input_users:
        for day_index, day in enumerate(tqdm(days)):
            times_in_that_day = generate_time_list(day, day+timedelta(days=1), bins, True)
            #print(times_in_that_day)
            for bin_index, times in enumerate(times_in_that_day):

                start = times[0].strftime("%Y-%m-%d %H:%M:%S")+timezones[country]
                end = times[1].strftime("%Y-%m-%d %H:%M:%S")+timezones[country]
                #print("start = {}, end = {}\nbin_index = {}\n-----------"
                #      .format(start, end, bin_index))
                result = d.run_specific_query("SELECT useruuid, count(*) FROM " +
                                              "location WHERE country='" + country +
                                              "' AND ((start_time >= '" + start +
                                              "' AND start_time < '" + end + "')" +
                                              " OR (end_time >= '" + start +
                                              "' AND end_time < '" + start + "')" +
                                              " OR (start_time < '"+start +
                                              "' AND end_time > '"+end+"'))" +
                                              "GROUP BY useruuid")
                for row in result:
                    user = row[0]
                    if counts:
                        data[users[user]][bin_index] += row[1]
                    else:
                        data[users[user]][bin_index] += 1
    else:
        for day_index, day in enumerate(tqdm(days)): #Vi starter på en mandag!!!
            times_in_that_day = generate_time_list(day, day+timedelta(days=1), bins, True)
            for bin_index, times in enumerate(times_in_that_day):
                for user in input_users:
                    start = times[0].strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                    end = times[1].strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                    result = d.run_specific_query("SELECT count(*) FROM " +
                                                  "location WHERE country='" + country +
                                                  "' AND (start_time >= '" + start +
                                                  "' AND start_time < '" + end + "')" +
                                                  "' OR (end_time >= '" + start +
                                                  "' AND end_time < '" + start + "')" +
                                                  " OR (start_time < '"+start +
                                                  "' AND end_time > '"+end+"')" +
                                                  " AND useruuid='"+user+"'")
                    for row in result:
                        if row[0] > 0:
                            if counts:
                                data[users[user]][bin_index] += row[0]
                            else:
                                data[users[user]][bin_index] += 1
                        
    return data, users


def generate_time_list(start_time, end_time, step_size, zips=False):
    """
    Generate a list of datetimes with an rrule of specific number of minutes

    Arguments:
        start_time {datetime} -- Datetime for when the list starts
        end_time   {datetime} -- Datetime for when the list ends
        step_size  {int}      -- How many minutes between element in the list (minutes - 60 = 1 hour)

    Return:
        list of datetimes
    """
    lst = []
    if not zips:
        lst.append(start_time)
        temp_time = start_time
        while temp_time < end_time:
            temp_time = temp_time+timedelta(minutes=step_size)
            lst.append(temp_time)
        lst = lst[:-1]
    else:
        temp_time = start_time
        before = temp_time
        while temp_time < end_time:
            temp_time = temp_time+timedelta(minutes=step_size)
            lst.append((before, temp_time))
            before = temp_time
    return lst



def week_of_month(date):
    """
    Calculate which week number of a month a date are in 
    That is between 1 and 6

    Arguments:
        date  {datetime} -- Datetime for which week number should be calculated for

    Returns:
        int
    """
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


def cdf_xy_plot():
    """
    Plotting the CDF for countries from fetch_data
    """
    _, df, _, countries = fetch_data()
    no_users = {"Japan": 316, "Sweden": 542}
    data = []
    for i, country in enumerate(countries):
        updates = list(df["Location updates"][df['country'] == country])
        c = dict(collections.OrderedDict(sorted(collections.Counter(updates).items())))
        x = []
        y = []
        for updates, users in sorted(c.items()):
            x.append(updates)
            y.append(users/no_users[country])
        data.append((x, np.cumsum(y)))
    
    title = "CDF for location updates"
    xlabel = "Location updates"
    ylabel = "Frequency"
    legend_labels = countries
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.plot(data[0][0], data[0][1], label=legend_labels[0], linewidth=6.0)
    plt.plot(data[1][0], data[1][1], label=legend_labels[1], color='r', linewidth=6.0)
    xlim_max = max([max(data[0][0]), max(data[1][0])])
    plt.ylim(0, 1)
    plt.xlim(0, xlim_max)
    plt.title(title)
    plt.legend(prop={'size': 40})
    sns.plt.tick_params(labelsize=20)
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()


def xy_plot(data, title, xlabel, ylabel):
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.plot(data[0], data[1])
    xlim_max = max(data[0])
    plt.ylim(0, max(data[1]))
    plt.xlim(0, xlim_max)
    plt.title(title)
    #plt.legend(prop={'size': 28})
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


def heat_map(data, mask, xticks, yticks, title="", xlabels=[], ylabels=[], anno=True, multiple=True, max_val=0):
    """
    Plotting a heatmap

    Arguments:
        data  {numpy array} -- Data to be plottet
        mask  {numpy array} -- Shows which cells to "remove" in the heatmap
        xlabels {list} -- One or more labels for X-axis
        ylabels {list} -- One or more labels for Y-axis
        titles  {string} -- title of heatmap
        anno    {boolean} -- Indicates whether the heatmap should show the values on map (True) or not (False)
        multiple {boolean} -- Indicates whether there are data for multiple heatmap (True) or not (False)
        max_val {int}     --  max value of the indicator. If it's 0 (default) max_val will be calculated from the data
    """
    sns.set(font_scale=2.5)
    if multiple:
        months = ["September", "October", "November"]
        if max_val == 0:
            max_val = np.amax(data)
        fig, ax = sns.plt.subplots(3, 1)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for index, ax in enumerate(ax.flat):
            sns.heatmap(data[index], ax=ax, xticklabels=[" "]*len(xticks[index]) if index != 2 else xticks[index],
                        annot=anno, fmt="d",
                        yticklabels=yticks[index], mask=mask[index], vmin=0,
                        vmax=max_val, cbar=index == 0, cbar_ax=None if index else cbar_ax)
            if title != "" and index == 0:
                ax.set_title(title)
            ax.set_ylabel(months[index])
            #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    else:
        if max_val == 0:
            max_val = np.amax(data)
        if mask:
            ax = sns.heatmap(data, xticklabels=xticks,
                             annot=anno, fmt="d",
                             yticklabels=yticks, mask=mask, vmin=0,
                             vmax=max_val)
        else:
            if not yticks:
                yticks = [[" "]*data.shape[1]]
            ax = sns.heatmap(data, xticklabels=xticks[0],
                             annot=anno, fmt="d",
                             yticklabels=yticks[0], vmin=0,
                             vmax=max_val)
            [label.set_visible(False) for label in ax.yaxis.get_ticklabels()]

            for label in ax.yaxis.get_ticklabels()[::4]:
                label.set_visible(True)
        ax.set_title(title)
        if ylabels:
            ax.set_ylabel(ylabels)
        else:
            ax.set_ylabel("Users")
        if xlabels:
            ax.set_xlabel(xlabels)
        else:
            ax.set_xlabel("Days")
        plt.yticks(rotation=0)
    sns.plt.show()


def boxplot(data, xx, yy, title):
    sns.set(font_scale=2.5)
    ax = sns.boxplot(x=xx, y=yy, data=data)
    ax.set_title(title)
    sns.plt.show()


def data_summary_per_user():
    """
    Calculates the quantiles for the data
    """
    _, df, total_counts, _, = fetch_data()
    q = df.groupby('country')['Location updates'].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    means = df.groupby('country')['Location updates'].mean()
    print("///////// Japan ////////")
    print(q['Japan'])
    print("////////")
    print("///////// Sweden //////////")
    print(q['Sweden'])
    print("////////")
    print("Japan - mean: {}".format(means['Japan']))
    print("Sweden - mean: {}".format(means['Sweden']))


def show_all_month_same_scale(countries, values_loc_updates=True,
                              specific_users=[],
                              sorter_non_location_user_fra=True,
                              sorter_efter_sum=True, find_users=False,
                              user_start_id=-1,
                              number_of_days_without_updates=5, anno=False,
                              base_title="Heatmap of location updates per users per day in "):

    """
    Plotting heatmaps of location updates per users per day in each month (sep, okt, nov) for each country. 
    The month are hardcoded

    Arguments:
        countries                      {list}    -- List of countries
        values_loc_updates             {boolean} -- Whether to get number of updates (True) or just whether there are or aren't an update (False)
        specific_users                 {list}    -- Only shows for certain users
        sorter_non_location_user_fra   {boolean} -- Whether to remove user who haven't any updates (True) or not (False)
        sorter_efter_sum               {boolean} -- Whether to sort the list of users after the sum of updates (True) or not (False)
        find_users                     {boolean} -- Whether to find users (True) or not (False)
        user_start_id                  {int}     -- User ID of the user to find users from (if find_users are True)
        number_of_days_without_updates {int}     -- A threshold for finding users (if find_users are True). 
                                                    It is max number of days of with the users can have no updates
        base_title                     {string}  -- (Base) title for the heatmaps

    """
    periods = [("2015-09-01", "2015-09-30"), ("2015-10-01", "2015-10-31"),
               ("2015-11-01", "2015-11-30")]
    max_values = defaultdict(dict)
    for country in countries:
        for fro, to in periods:
            data, users = get_data_for_heat_map_per_user(country, fro, to,
                                                         specific_users,
                                                         values_loc_updates)
            max_value = np.amax(data)
            if country in max_values:
                if max_value > max_values[country]:
                    max_values[country] = max_value   
            else:
                max_values[country] = max_value
    for country in countries:
        for fro, to in periods:
            show_specific_country_and_period(country, fro, to,
                                             values_loc_updates, base_title,
                                             specific_users,
                                             sorter_non_location_user_fra,
                                             sorter_efter_sum, find_users,
                                             user_start_id,
                                             number_of_days_without_updates,
                                             max_values[country], anno)


def show_specific_country_and_period(country, fro, to, values_loc_updates,
                                     base_title, specific_users=[],
                                     sorter_non_location_user_fra=True,
                                     sorter_efter_sum=True, find_users=False,
                                     user_start_id=-1,
                                     number_of_days_without_updates=5,
                                     max_val=0, anno=False):
    """
    Plotting a single heatmap of location updates per users per day in a month 
    The month are hardcoded

    Arguments:
        country                        {string}  -- Country
        fro                            {string}  -- From datetime string
        to                             {string}  -- To datetime string
        values_loc_updates             {boolean} -- Whether to get number of updates (True) or just whether there are or aren't an update (False)
        base_title                     {string}  -- (Base) title for the heatmap
        specific_users                 {list}    -- Only shows for certain users
        sorter_non_location_user_fra   {boolean} -- Whether to remove user who haven't any updates (True) or not (False)
        sorter_efter_sum               {boolean} -- Whether to sort the list of users after the sum of updates (True) or not (False)
        find_users                     {boolean} -- Whether to find users (True) or not (False)
        user_start_id                  {int}     -- User ID of the user to find users from (if find_users are True)
        number_of_days_without_updates {int}     -- A threshold for finding users (if find_users are True). 
                                                    It is max number of days of with the users can have no updates
        max_val                        {int}     -- Max values for the scale in heatmap
        anno                           {boolean} -- Whether to show values in the cells of the heatmap (True) or not (False)

    """
    data, users = get_data_for_heat_map_per_user(country, fro, to,
                                                 specific_users,
                                                 values_loc_updates)
    ylabels = []
    if sorter_non_location_user_fra:
        for i, x in enumerate(np.any(data != 0, axis=1)): #Sorter dem der har rene 0 fra
            if x:
                ylabels.append(i)
        data = data[np.any(data != 0, axis=1)]
    else:
        ylabels = list(range(data.shape[0]))
    if sorter_efter_sum:
        s = np.sum(data, axis=1)
        data = np.take(data, s.argsort(), axis=0)
        ylabels = np.take(ylabels, s.argsort(), axis=0)
    #Find users...
    if find_users:
        if user_start_id == -1:
            start = 0
        else:
            start = list(ylabels).index(user_start_id) #sverige: nov: 270, okt: 293, sep: 475, - japan: #sep: 243, now:14, okt:226
        found_users = []
        for i, row in enumerate(data[start:], start=start):
            res = collections.Counter(row)
            print(res)
            if 0 in res:
                if res[0] < number_of_days_without_updates: #sverige: nov: 6, okt: 3, sep: 6 - japan: #sep: 5, nov: 0, okt: 3
                    found_users.append(users[ylabels[i]])
                else:
                    if specific_users:
                        if users[ylabels[i]] in specific_users:
                            print("user er i users_input")
                        else:
                            print("nej, user er ikke i specific_users")
            else:
                found_users.append(users[ylabels[i]])
        print("find_users\n--------------")
        print("Antal found_users: {}\n{}".format(len(found_users),
                                                 found_users))
    ylabels = [ylabels]
    idxs = find_duplicate_rows(data)
    for group in idxs:
        print("#### Ny gruppe der er ens! ####")
        for user in set(group):
            print(users[ylabels[0][user]])
        print("#######################")
    xlabels = [list(range(1, data.shape[1]+1))]
    heat_map(data, [], xlabels, ylabels, title=base_title+country +
             " from "+fro+" to "+to, anno=anno, multiple=False,
             max_val=max_val)


def show_all_month_for_contries():
    """
    Show a heatmap of the 3 months (sep, okt, nov) for each country
    """
    data, df, total_count, countries = fetch_data()

    #............... HEAT-MAP --------------------#
    for country in countries:
        user = ""
        data, mask = get_data_for_heat_map_per_month(country, total_count)
        title = "Number of location updates per user in "+country
        if user != "":
            title += " for user " + str(user)
        ylabels = [["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"]]
        xlabels = [["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]]
        heat_map(data, mask, xlabels, ylabels, title, True)


def find_duplicate_rows(data):
    """
    find rows in a sorted numpy array of 0 and 1,
    where columns are equal in 90% or more of the cases

    Arguments:
        data    {numpy}  -- Numpy array with 0 or 1's

    Return:
        list of list with indices of the rows which are flagged as duplicates
    """
    length = data.shape[0]
    threshold = math.ceil(data.shape[1]*0.9)
    zero_threshold = math.ceil(data.shape[1]*0.5)
    lst = []
    temp_lst = []
    for i, d in enumerate(data, start=1):
        if i < length:
            row = collections.Counter(d)
            if 0 not in row or row[0] <= zero_threshold:
                res = np.bincount(d == data[i])
                if res.shape[0] > 1: #Der er nogle der er true
                    if res[1] >= threshold:
                        temp_lst.extend([i-1, i])
                    else:
                        if temp_lst:
                            lst.append(temp_lst)
                            temp_lst = []
                else:
                    if temp_lst:
                        lst.append(temp_lst)
                        temp_lst = []
            else:
                if temp_lst:
                    lst.append(temp_lst)
                    temp_lst = []
    return lst


def compare_loc_updates_per_month():
    """
    Plots a plot  of mean of location updates of Japan and Sweden
    """
    d = DatabaseHelper.DatabaseHelper()
    countries = ["Japan", "Sweden"]
    no_users = {"Japan": 316, "Sweden": 542}
    periods = [("2015-09-01", "2015-10-01"), ("2015-10-01", "2015-11-01"),
               ("2015-11-01", "2015-12-01")]
    months = ["September", "October", "November"]
    dd = {'Countries': [], 'Month': []}
    for i, period in enumerate(periods):
        for country in countries:
            query = "SELECT count(*)/"+str(no_users[country])+" FROM location WHERE country='"+country+"'" +\
                    " AND start_time >= '"+period[0] + "' " +\
                    "AND start_time < '"+period[1] + "'"
            result = d.run_specific_query(query)
            #print(result[0])
            for row in result:
                [dd["Countries"].append(country) for x in range(row[0])]
                [dd["Month"].append(months[i]) for x in range(row[0])]

    df = pd.DataFrame(dd)
    #print(df)
    sns.set(font_scale=2.5)
    ax = sns.countplot(x="Month", hue="Countries", data=df)
    ax.set_ylabel("Number of location updates")
    ax.set_title("Mean of location updates in Japan and Sweden over three month period")
    #ax.set_xlabel("Days")
    sns.plt.legend(prop={'size': 40})
    sns.plt.tick_params(labelsize=20)
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    sns.plt.show()


def fetch_data(countries=["Japan", "Sweden"]):
    """
    Arguments:
        countries    {countries}  -- Which countries to fetch data for

    Return:
        data         {list}       -- List raw data
        df           {dataframe}  -- Pandas dataframe of data
        total_count  {dict}       -- dicts with country as keys and number of distinct users as values
        countries    {list}       -- List of countries
    """
    data = []
    titles = []
    labels = []
    xlabels = []
    d = DatabaseHelper.DatabaseHelper()

    dd = {'Location updates': [], 'country': []}
    total_count = {}
    for i, country in enumerate(countries):
        query = "SELECT useruuid, count(*) FROM location WHERE country='"+country+"'"+ \
                "GROUP BY useruuid ORDER BY useruuid"
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
    df = pd.DataFrame(dd)
    return data, df, total_count, countries


def location_updates_at_hq_or_not():
    d = DatabaseHelper.DatabaseHelper()
    filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                    [(17.9529121, 59.4050982), 1000]],
                          "Japan": [[(139.743862, 35.630338), 1000]]}
    data, df, total_count, countries = fetch_data()

    test = {'Country': [], 'hq': []}
    #test = pd.DataFrame([], columns=['country', 'hq'])
    temp_i = 0
    for country in countries:
        print(country)
        for row in df.loc[(df['country'] == country), 'Location updates']:
            test['Country'].extend([country]*row)
            test['hq'].extend(['At HQ']*row)
    test2 = pd.DataFrame(test)
    for country in countries:
        res = d.get_locations_for_numpy(filter_places_dict[country])
        res = [row for row in res if row[3] == country]
        if country == 'Sweden':
            print(len(res))
            test2.loc[(test2['Country'] == country) & (test2.index < (299157+len(res))), 'hq'] = 'Not at HQ'
        else:
            test2.loc[(test2['Country'] == country) & (test2.index < len(res)), 'hq'] = 'Not at HQ'
    test3 = {'total': [], 'Not at hq': [], 'Country': []}
    for i, country in enumerate(countries):
        test3['Country'].append(country)
        test3['total'].append(test2.loc[(test2['Country'] == country)].shape[0])
        test3['Not at hq'].append(test2.loc[(test2['Country'] == country) &
                                            (test2['hq'] != 'At HQ')].shape[0])
    test4 = pd.DataFrame(test3)
    sns.set(font_scale=2.5)

    #
    ax = sns.barplot(x="Country", y='total', data=test4, color='#e74c3c')#, color="red"
    bottom_plot = sns.barplot(x="Country", y="Not at hq", data=test4, color='#3498db') #color="#0000A3"  #'#3498db'

    bottombar = plt.Rectangle((0, 0), 1, 1, fc='#3498db',  edgecolor='none')
    topbar = plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", edgecolor='none')
    
    l = plt.legend([bottombar, topbar], ['Outside perimeters', 'Inside perimeters'], loc=1, ncol=2, prop={'size': 40})
    l.draw_frame(False)

    bottom_plot.set_ylabel("Number of location updates")
    bottom_plot.set_title("Distribution of location updates (inside/outside Sony perimeters)")
    #ax.set_xlabel("Days")
    #plt.legend(prop={'size': 40})
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.2e'))
    sns.plt.tick_params(labelsize=20)
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    sns.plt.show()


def coocs_at_hq_or_not():
    d = DatabaseHelper.DatabaseHelper()
    ds = DatasetHelper.DatasetHelper()
    filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                    [(17.9529121, 59.4050982), 1000]],
                          "Japan": [[(139.743862, 35.630338), 1000]]}

    countries = ["Japan", "Sweden"]
    result = {'total': [], 'Not at hq': [], 'Country': []}
    _, countries_dict, locations = d.generate_numpy_matrix_from_database()
    for country in countries:
        result['Country'].append(country)
        coocs = ds.generate_cooccurrences_array(locations[locations[:, 3] ==
                                                countries_dict[country]])
        result['total'].append(coocs.shape[0])
        _, countries_dict2, locations2 = \
            d.generate_numpy_matrix_from_database(filter_places_dict[country])
        coocs2 = ds.generate_cooccurrences_array(locations2[locations2[:, 3] ==
                                                 countries_dict2[country]])
        result['Not at hq'].append(coocs2.shape[0])

    test4 = pd.DataFrame(result)
    print(test4)
    sns.set(font_scale=2.5)

    #
    ax = sns.barplot(x="Country", y='total', data=test4, color='#e74c3c')#, color="red"
    bottom_plot = sns.barplot(x="Country", y="Not at hq", data=test4, color='#3498db') #color="#0000A3"  #'#3498db'

    bottombar = plt.Rectangle((0, 0), 1, 1, fc='#3498db',  edgecolor='none')
    topbar = plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", edgecolor='none')
    
    l = plt.legend([bottombar, topbar], ['Outside perimeters', 'Inside perimeters'], loc=1, ncol=2, prop={'size': 40})
    l.draw_frame(False)

    bottom_plot.set_ylabel("Number of co-occurrences")
    bottom_plot.set_title("Distribution of co-occurrences (inside/outside Sony perimeters)")
    #ax.set_xlabel("Days")
    #plt.legend(prop={'size': 40})
    sns.plt.ylim(0, 1300000)
    sns.plt.tick_params(labelsize=20)
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.2e'))
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    sns.plt.show()


def time_plot():
    d = DatabaseHelper.DatabaseHelper()
    data = []
    _, _, total_count, countries = fetch_data()
    for country in countries:
        x = []
        y = []
        res = d.run_specific_query("select extract(hour FROM start_time), count(*) from location WHERE country='"+country+"' GROUP BY extract(hour FROM start_time)")
        for row in res:
            x.append(adjust_hour(row[0], country))
            y.append(row[1]/total_count[country])
        temp = zip(x, y)
        temp2 = list(zip(*sorted(temp)))
        data.append((list(temp2[0]), list(temp2[1])))
    title = "Cummulative location updates by time of day"
    xlabel = "Time of day in hours"
    ylabel = "Cummulative location updates frequency"
    legend_labels = countries
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.plot(data[0][0], data[0][1], label=legend_labels[0], linewidth=6.0)
    plt.plot(data[1][0], data[1][1], label=legend_labels[1], color='r', linewidth=6.0)
    #xlim_max = max([max(data[0][0]), max(data[1][0])])
    #plt.ylim(0, 1)
    plt.xticks(data[0][0])
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
    plt.xlim(0, 24)
    plt.title(title)
    plt.legend(prop={'size': 40})
    sns.plt.tick_params(labelsize=20)
    [item.set_fontsize(45) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(48)
    [item.set_fontsize(33) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()


def adjust_hour(hour, country):
    countries = {'Japan': 7, 'Sweden': 0}
    return (hour+countries[country]) % 24
    

def aggregated_heatmap():
    country = "Sweden"
    start = '2015-09-01'
    end = '2015-11-30'
    bin_size = 60 #minutes
    data, users = get_data_for_heat_map_per_user_aggregated(country, start, end, bin_size)
    lst = generate_time_list(parser.parse(start+' 00:00:00+00:00'),
                             parser.parse(start+' 00:00:00+00:00') +
                             timedelta(days=1), bin_size)
    xticks = [row.strftime("%H:%M") for row in lst]
    print(xticks)
    yticks = list(range(data.shape[0]))
    heat_map(data, [], [xticks], [],
             title=country + " - '" + start + "' to '" + end + "'",
             xlabels="Time of day", ylabels=[], anno=False, multiple=False, max_val=0)


if __name__ == '__main__':
    aggregated_heatmap()
    #res = generate_time_list(parser.parse('2015-08-30 00:00:00+00:00'),
    #                         parser.parse('2015-08-31 00:00:00+00:00'), 60, True)
    #print(len(res))
    #print(res)
    
    #coocs_at_hq_or_not()
    #location_updates_at_hq_or_not()
    #cdf_xy_plot()
    #time_plot()
    #compare_loc_updates_per_month()
    #data, df, total_count, countries = fetch_data()

    #show_all_month_for_contries()
    #cdf_xy_plot()
    #print("show_all_month_same_scale")
    #show_all_month_same_scale(countries, find_users=False,
    #                          number_of_days_without_updates=5)
    #print("show_specific_country_and_period")
    country = "Japan"
    fro = "2015-11-01"
    to = "2015-11-30"
    #show_specific_country_and_period(country, fro, to, False,
    #                                 base_title="Heatmap of location updates per users\
    #                                             per day in ")




