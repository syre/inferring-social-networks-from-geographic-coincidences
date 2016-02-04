#!/usr/bin/env python3.5
from collections import defaultdict
import json

import matplotlib.pyplot as plt  
import numpy as np

#all_data = []
country_data = defaultdict(dict)
country_data['countries'] = set()



def open_file(filename):
    with open(filename) as json_file:
        d = json.load(json_file)
    return d

all_data = open_file("all_201511.json")

#print(len(all_data))
for data in all_data: 
    #print(data)
    if data['country'] != '': #
        country_data['countries'].add(data['country'])
        if data['country'] in country_data['country_numbers']:
            country_data['country_numbers'][data['country']] += 1
        else: 
            country_data['country_numbers'][data['country']] = 1



print("Number of countries: {0}".format(len(country_data['countries'])))
#print("Countries: {0}".format(country_data['countries']))

#for country in country_data['countries']:
#    print("Occurrences of {0}: {1}".format(country, country_data['country_numbers'][country]))

countries_list = []
numbers = []
for country in country_data['countries']:
    countries_list.append(country)
    numbers.append(country_data['country_numbers'][country])

countries_list_alph, numbers_alph = zip(*sorted(zip(countries_list, numbers)))
numbers_asc, countries_list_asc = zip(*sorted(zip(numbers, countries_list)))
numbers_desc, countries_list_desc = zip(*sorted(zip(numbers, countries_list), reverse=True))


total_geotags = np.sum(numbers)
print("total_geotags = {0}, numbers_desc[0] = {1}, countries_list_desc[0] = {2}".format(total_geotags, numbers_desc[0],countries_list_desc[0]))
print("{0} stands for {1}% of all geotagging".format(countries_list_desc[0], ((numbers_desc[0]/total_geotags)*100)))


def normalize(numbers):
    total = np.sum(numbers)
    norm_numbers = []
    for n in numbers:  #Normalize numbers
        norm_numbers.append(n/total)
    return norm_numbers


def calc_cdf(numbers):
    return np.cumsum(normalize(numbers))  #Return the cumulative numbers



def plot_cdf(plot_type, numbers, title, x_label, y_label, x_ticks=[], y_ticks=[]):
    x = np.arange(len(countries_list))
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(12, 14))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
      
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    


    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on") 

    ind = np.arange(len(countries_list))  # the x locations for the groups
    width = 0.35       # the width of the bars

    if plot_type == "barplot":
        ind = np.arange(len(countries_list))  # the x locations for the groups
        width = 0.35       # the width of the bars
        #rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)
        plt.bar(ind+width, numbers, width)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_ticks != []:
            plt.xticks(ind + width*1.5, x_ticks, rotation='vertical')
        if y_ticks != []:
            plt.yticks(ind + width*1.5, y_ticks, rotation='vertical')
        plt.show()
    
    elif plot_type == "xy":
        plt.plot(x, numbers)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_ticks != []:
            plt.xticks(ind + width*1.5, x_ticks, rotation='vertical')
        if y_ticks != []:
            plt.yticks(ind + width*1.5, y_ticks, rotation='vertical')
        plt.show() 

    else:
        print("Unknown plot-type!!!")               



cdf = calc_cdf(numbers_asc)

plot_cdf("xy", cdf, "CDF of geotagging over countries", "Countries", "CDF values", countries_list_asc)


def plot_barplot(numbers, title, x_label, y_label, x_ticks=[], y_ticks=[]):
    ind = np.arange(len(x_ticks))  # the x locations for the groups
    width = 0.35       # the width of the bars

    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(12, 14))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
      
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    


    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on") 

    #rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)
    plt.bar(ind + width, numbers, width)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_ticks != []:
        plt.xticks(ind + width*1.5, x_ticks, rotation='vertical')
    if y_ticks != []:
        plt.yticks(ind + width*1.5, y_ticks, rotation='vertical')
    plt.show()          

plot_barplot(numbers_alph, "Distribution of geotagging in countries", "Countries", "Number of geotagging", countries_list_alph)

def get_procentile_of_data(data_numbers, labels, procent):
    i = 0;
    data_numbers_cdf = calc_cdf(data_numbers)
    data_numbers_norm = normalize(data_numbers)
    for n in data_numbers_cdf:
        if n>=procent:
            break
        i+=1
    return data_numbers_norm[:i+1], labels[:i+1]

procent = 0.9
numbers09, countries09 = get_procentile_of_data(numbers_desc, countries_list_desc, procent)
print("Countries which represent minimum {0}% of data:".format(procent*100))
index = 0
for n in numbers09:
    print("{0}: {1}%".format(countries09[index], n*100))
    index +=1
print("Total: {0}%".format(np.sum(numbers09)*100))