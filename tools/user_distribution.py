#!/usr/bin/env python3.5

from Distribution import Distribution
import numpy as np

dist = Distribution()

files = ["all_201511.json","all_201510.json","all_201509.json"]
keys = ["useruuid"]
req_keys = ["country"]
req_vals = ["Japan"]
labels_alph, numbers_alph, labels_asc, numbers_asc, labels_desc, numbers_desc = dist.fetch_data(files, keys, req_keys, req_vals)
cdfs = []
for nums in numbers_asc:
    cumsum = dist.calc_cdf(nums)
    cdfs.append(cumsum)

titles = []
x_labels = []
y_labels = []
x_ticks = []
y_ticks = []
index = 0
for _ in numbers_desc:
    titles.append("CDF of geotagging over countries for ["+files[index]+"]")
    x_labels.append("Useruuid")
    y_labels.append("CDF values")
    x_ticks.append([])
    y_ticks.append([])
    index+=1



dist.plot_cdf("xy", cdfs, titles, x_labels, y_labels, x_ticks,  y_ticks)

titles = []
x_labels = []
y_labels = []
x_ticks = []
y_ticks = []
index = 0
for _ in numbers_asc:
    titles.append("Distribution of geotagging in useruuid for ["+files[index]+"]")
    x_labels.append("Useruuid")
    y_labels.append("Number of geotagging")
    x_ticks.append([])
    y_ticks.append([])
    index+=1

dist.plot_barplot_subplot(numbers_asc, titles, x_labels, y_labels, x_ticks, y_ticks)


print("Top 5 in each file:\n-------------")
index = 0
for f in files:
    print(f+":\n-------")
    for n in numbers_desc[index][:5]:
        print(n)
    index+=1
