#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import FileLoader
from collections import Counter
import seaborn as sns
import DatabaseHelper
import pprint
from matplotlib import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck



def get_data():
    country_dist = d.get_distributions_categories("country")
    countries = []
    data = []
    filter_countries = ["Sweden", "Japan", "United States", "China", "United Kingdom"]
    for row in tqdm(country_dist['results']):
        #countries.append(row['Countries'])
        #values.append(row['Count'])
        if row['Countries'] in filter_countries:
            data.extend([row['Countries']]*row['Count'])
            countries.append(row['Countries'])
    return data, countries


def bar_plot(data, labels):
    test = {'test2': data}

    #fig, ax = sns.plt.subplots()
    # the size of A4 paper
    #fig.set_size_inches(11.7, 8.27)
    sns.set(font_scale=4.5)
    ax = sns.countplot(x="test2", data=test)
    #sns.set(font_scale=2.5)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Countries") #fontsize=35
    ax.set_ylabel("Number of location updates") #, fontsize=35
    #ax.set(ylabel="Count")
    ax.set(title="Country distribution")
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.2e'))
    #sns.plt.tick_params(labelsize=28)
    #sns.plt.rcParams['axes.labelsize'] = 25
    #sns.plt.rcParams['legend.fontsize'] = 18
    #sns.plt.tight_layout()
    sns.plt.show()
    


if __name__ == '__main__':
    d = DatabaseHelper.DatabaseHelper()
    #print(sns.load_dataset("titanic"))
    data, labels = get_data()

    bar_plot(data, labels)
    
    
