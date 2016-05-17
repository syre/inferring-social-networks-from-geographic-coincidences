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
import collections
import pprint


def get_data_for_heat_map_per_month(country="Japan", total_count={}, user=""):
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
                    data[month_index][day.weekday()][week_of_month(day)-1] = result/total_count[country]
                    
                else: #Hvis vi er i current_month
                    if day.month != 8:
                        start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                        end =  day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                        user_query = ""
                        if user != "":
                            user_query = " AND useruuid = '"+user+"'"
                        query = "select count(*) FROM location WHERE country='"+country+"' AND start_time >= '"+start+"' AND start_time <= '" + end + "'" + user_query+";"
                        result = d.run_specific_query(query)[0][0] 
                        data[month_index][day.weekday()][week_of_month(day)-1] = result/total_count[country]
        if break_flag:
            break 

    print(data)
    mask = data == -1
    print(mask)
    return data, mask


def get_all_users_in_country(country, users=[]):
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

    if not input_users:
        for day_index, day in enumerate(tqdm(days)): #Vi starter på en mandag!!!
            start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
            end =  day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
            result = d.run_specific_query("SELECT useruuid, count(*) FROM " +
                                          "location WHERE country='" + country +
                                          "' AND start_time >= '" + start +
                                          "' AND start_time <= '" + end + "'" +
                                          "GROUP BY useruuid")
            for row in result:
                user = row[0]
                if counts:
                    data[users[user]][day_index] = row[1]
                else:
                    data[users[user]][day_index] = 1
                #counts = row[1]
                
        #print(data)
    else:
        for day_index, day in enumerate(tqdm(days)): #Vi starter på en mandag!!!
            for user in input_users:
                start = day.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
                end = day.strftime("%Y-%m-%d")+" 23:59:59+00:00"
                result = d.run_specific_query("SELECT count(*) FROM " +
                                              "location WHERE country='" + country +
                                              "' AND start_time >= '" + start +
                                              "' AND start_time <= '" + end + "'" +
                                              " AND useruuid='"+user+"'")
                for row in result:
                    if row[0] > 0:
                        if counts:
                            data[users[user]][day_index] = row[0]
                        else:
                            data[users[user]][day_index] = 1
                        
    return data, users


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
    
    xlim_max = max([max(data[0][0]), max(data[1][0])])
    plt.ylim(0, 1)
    plt.xlim(0, xlim_max)
    plt.title(title)
    plt.legend(prop={'size': 28})
    [item.set_fontsize(35) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(40)
    [item.set_fontsize(28) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    plt.show()


def xy_plot(data, title, xlabel, ylabel):
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.plot(data[0], data[1])
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    
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


def heat_map(data, mask, xlabels, ylabels, title="", anno=True, multiple=True, max_val=0):
    sns.set(font_scale=2.5)
    if multiple:
        months = ["September", "October", "November"]
        if max_val == 0:
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
    else:
        if max_val == 0:
            max_val = np.amax(data)
        if mask:
            ax = sns.heatmap(data, xticklabels=xlabels,
                             annot=anno, fmt="d",
                             yticklabels=ylabels, mask=mask, vmin=0,
                             vmax=max_val)
        else:
            if not ylabels:
                ylabels = [[" "]*data.shape[1]]
            ax = sns.heatmap(data, xticklabels=xlabels[0],
                             annot=anno, fmt="d",
                             yticklabels=ylabels[0], vmin=0,
                             vmax=max_val)
            [label.set_visible(False) for label in ax.yaxis.get_ticklabels()]

            for label in ax.yaxis.get_ticklabels()[::4]:
                label.set_visible(True)
        ax.set_title(title)
        ax.set_ylabel("Users")
        ax.set_xlabel("Days")
        plt.yticks(rotation=0)
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


def find_duplicate_rows(data):
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
    sns.plt.tick_params(labelsize=20)
    [item.set_fontsize(35) for item in [ax.yaxis.label, ax.xaxis.label]]
    ax.title.set_fontsize(40)
    [item.set_fontsize(28) for item in ax.get_xticklabels() + ax.get_yticklabels()]
    sns.plt.show()

if __name__ == '__main__':
    #compare_loc_updates_per_month()
    #sys.exit(0)
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
        #data, mask = get_data_for_heat_map_per_month(country, total_count)
        title = "Number of location updates per user in "+country
        if user != "":
            title += " for user " + str(user)
        ylabels = [["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"],
                   ["M", "T", "W", "T", "F", "S", "S"]]
        xlabels = [["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]]
        #heat_map(data, mask, xlabels, ylabels, title, True)
    
    
    
    #users_input = ['38da5e71-0062-45a7-8021-f90680260b61', '4ed45839-d8a7-40df-9c93-2554201f62ca', '458f17fa-126d-4079-a2d3-058a9ce2f57c', '69935fdf-9b43-417f-93df-a6f707e8b43f', 'd5a6b8e9-8ae5-4e9d-94d5-2c1865ad2e44', '4b19bbd7-0df5-4ae8-929d-ef3eae78fdb8', '4ccc2f16-5a33-4fca-a644-dc5c75a3deaa', 'b992b237-e563-48d7-b958-2b3e16620846', '3084b64d-e773-4daa-aeea-cc3b069594f3', '463b7dbb-ef29-49d0-a240-41baceea128f', '6255db24-5443-40d3-b65a-ae1b11288c8a', 'c6309812-9802-4b93-8e5d-14b5eb738438', '489e1e7c-c208-4601-b56a-287981417efe', '45691b60-e48d-4c08-9547-26223bdd7134', '0c5552ca-09c6-4cbf-ae0c-5969c3ad9983', '01caa88b-fa3c-4b03-b3e9-2b7a14beb278', 'eb34bbe5-fa09-42f6-a411-cdb7d3a29a20', 'ac9ff428-168b-432e-a6ae-f5571c3711bf', '01959ac9-8909-4d95-9557-f5b531a7f331', '2abf30d9-55f6-454e-854d-786f037b619c', '26daf874-26f3-4dfc-8679-9c15aecbc18d', 'e1fa02ad-4d6d-4def-86a2-1ade8c59ee8e', '0a21780a-1869-4cec-8ccf-50f57f5c7797', '8480e4ca-f1b8-449a-b571-5b3f3cd93e4d', 'aba7356f-77e9-4b57-8600-65f0fd479e10', 'ccd610d3-b349-44c2-9837-6b8df74d6fd1', '5a546365-f979-4f2c-a425-bddd93e667ee', '48292fe8-5d65-47f6-8c73-c23aa7030e03', '9b3edd01-b821-40c9-9f75-10cb32aa14b6', '64b6eb39-dd10-4a38-9bca-f0c90919b14c', '4992ac55-6d61-4e70-92f6-46549205f3bf', '8e9e90bc-1817-48d5-a4c8-1e3c4637262a', 'f4f32177-8c2c-418d-834c-0cde35f40cee', 'f23fd74a-f94f-4182-abaa-0ab8fb3e4a4f', '071c08fe-3fe2-49c6-ac1f-21f93cd1a87b', '7a749f71-6ccc-45bd-b4a2-782d3eb995ba', '66d1ac8f-7d29-4d2f-a241-e6bcb2c38489', 'fcc9196f-b23c-4839-b1c0-a853e3b35c8b', '163459e1-e9fd-4531-9382-b863a49adbf0', '6ce6542e-e348-454c-9bcc-f9e172e860ee', '1274307c-cd49-4f69-9cfd-9a598c426cfb', '6d4e2221-8099-4ef6-9e96-740963f74983', '6d20e7db-78f8-484c-bbf3-b5ae1f7e6b3b', '6c71df2e-54cb-489e-b110-8bb3234cfa0e', '0c4d0349-eb16-46df-9f33-4864a6717037', 'f509d570-c7c7-4cb3-9b86-b02836cac466', '0f8dd08e-7eb5-4614-aad5-cd4f9723f79c', '0d8bca9b-a4e6-4668-ba43-9521b6cb4f1e', '018b3a42-6ad5-4577-b2d8-341159eff9be', '105b7c5e-9d8a-4f65-b1d2-82bfa1e126e9', '9a9b1ef8-c48b-4105-b4fe-ab04acd3d0ee', 'e2215ac7-2be2-45f6-ae6e-149f89a8e7f2', '4bd3f3b1-791f-44be-8c52-0fd2195c4e62', '9df83008-ad97-4f83-a373-98fbb5b45ef2', '492f0a67-9a2c-40b8-8f0a-730db06abf65', 'a12f171b-76d9-4b58-a04f-9833bf10d2a3', 'a1eed029-bf59-48d1-919d-5689f6523426', 'a21d8502-c5cb-4eb5-8b19-b38ea74e9294', 'a221e5b4-ce58-49ba-9385-6b544cc8a5ae', 'a6c0e224-508b-4c28-a549-b85bd40ab770', 'aa360897-b5ab-431e-afc2-8bcbc7f484a6', 'ab65c6e5-5ee9-45d0-a352-a52c7ef9d9d6', '420b91b6-5c78-4fe6-af7c-9795edd10c0e', '3ee183a4-919a-4c67-bf84-6f196897a906', '3b48d329-f8c3-4fed-adcf-ef0aa81bbf2f', 'b0e1364a-9f2e-4b2b-8125-bd6176c16384', 'e3a58b79-b39f-46c0-8ba6-9836030ee133', '963d56e9-c390-49ad-ba83-d4c6574f676a', '36866473-4b75-4a52-a5a5-65ca6326aa04', '90522cae-ba2e-4677-a6e8-04ab2ddb64ed', '30fe0f1f-296f-4d4b-a4ef-93eaf8e05169', '8d325d9f-9341-4d00-a890-2adaf412e5ca', '8d03174e-311a-4b9f-9863-7d99f4fccd57', 'cc38817a-eb2c-448d-9194-595e210543f4', '2ddb668d-0c98-4258-844e-7e790ea65aba', '543036c6-93ab-40a6-9668-20a70d021cdf', '994b3bcf-a892-493e-b9f7-8477dec24cd5', 'd6e5fa22-27dd-47cb-acbe-eae33d029ae3', '29ec0b2f-dcc9-43e2-9421-ddada03513dd', 'd81f57bb-0925-4869-b844-8d99bf55337c', '5909445d-e68c-451f-baee-d108ca32c8cd', 'dffcd105-5c2e-41ed-848f-a20495571642', 'b3124440-b2f8-4add-9e67-ad6adf4ec501', '37abb413-d176-423c-8db6-61f253324c28']
    max_values = defaultdict(dict)
    #print("antal users: {}".format(len(users_input)))

    #print("show_all_month_same_scale")
    #show_all_month_same_scale(countries, find_users=True,
    #                          number_of_days_without_updates=5)

    #print("show_specific_country_and_period")
    country = "Japan"
    fro = "2015-11-01"
    to = "2015-11-30"
    #show_specific_country_and_period(country, fro, to, False,
    #                                 base_title="Heatmap of location updates per users\
    #                                             per day in ")

    
    #y_data = np.sum(data, axis=0)
    #x_data = xlabels[0]
    #rint(x_data)
    #print(y_data)

    #xy_plot([x_data, y_data], "Number of users with location updates per day for Japan", "Days", "Number of users with location updates")





