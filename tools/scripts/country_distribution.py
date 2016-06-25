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



def get_data(top_five=True):
    country_dist = d.get_distributions_categories("country")
    countries = []
    data = []
    filter_countries = ["Sweden", "Japan", "United States", "China", "United Kingdom"]
    for row in tqdm(country_dist['results']):
        if top_five:
            if row['Countries'] in filter_countries:
                data.extend([row['Countries']]*row['Count'])
                countries.append(row['Countries'])
        else:
            data.extend([row['Countries']]*row['Count'])
            countries.append(row['Countries'])
    return data, countries


def bar_plot(data, labels):
    test = {'test2': data}
    sns.set(font_scale=4.5)
    ax = sns.countplot(x="test2", data=test)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of location updates")
    ax.set(title="Top 5 countries with most location updates")
    sns.plt.show()
    

def func(x, pos):
    return str(int(x/1000))+"K"


if __name__ == '__main__':
    d = DatabaseHelper.DatabaseHelper()
    data, labels = get_data(top_five=True)

    bar_plot(data, labels)
    
    
