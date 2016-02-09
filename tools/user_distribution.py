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
    #print(cumsum[:-10])

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
#plot_type, numbers, labels, titles, x_labels, y_labels, x_ticks=[], y_ticks=[]

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
#plot_barplot(numbers_alph, "Distribution of geotagging in countries", "Countries", "Number of geotagging", key_data_list_alph)
#print(len(labels_alph[0]))


#print(numbers_desc[0][:10])
dist.plot_barplot_subplot(numbers_asc, titles, x_labels, y_labels, x_ticks, y_ticks)

print("Top 5 in each file:\n-------------")
index = 0
for f in files:
    print(f+":\n-------")
    for n in numbers_desc[index][:5]:
        print(n)
    index+=1

# procent = 0.95
# numbers09, countries09 = dist.get_procentile_of_data(numbers_desc[0], labels_desc[0], procent)
# print("Users which represent minimum {0}% of data:".format(procent*100))
# index = 0
# for n in numbers09:
#     print("{0}: {1}%".format(countries09[index], n*100))
#     index +=1
# print("Total: {0}%".format(np.sum(numbers09)*100))

