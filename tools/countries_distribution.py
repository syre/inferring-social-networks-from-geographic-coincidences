#!/usr/bin/env python3.5

from Distribution import Distribution

dist = Distribution()

files = ["all_201511.json","all_201510.json","all_201509.json"]
labels_alph, numbers_alph, labels_asc, numbers_asc, labels_desc, numbers_desc = dist.fetch_data(files, "country")
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
    x_labels.append("Countries")
    y_labels.append("CDF values")
    x_ticks.append(labels_asc[index])
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
for _ in numbers_alph:
    titles.append("Distribution of geotagging in countries for ["+files[index]+"]")
    x_labels.append("Countries")
    y_labels.append("Number of geotagging")
    #x_ticks.append(list_of_key_data_list_alph[index])
    y_ticks.append([])
    index+=1
#plot_barplot(numbers_alph, "Distribution of geotagging in countries", "Countries", "Number of geotagging", key_data_list_alph)
print(len(labels_alph[0]))

dist.plot_barplot_subplot(numbers_alph, titles, x_labels, y_labels, labels_alph, y_ticks)


# procent = 0.9
# numbers09, countries09 = get_procentile_of_data(numbers_desc, key_data_list_desc, procent)
# print("Countries which represent minimum {0}% of data:".format(procent*100))
# index = 0
# for n in numbers09:
#     print("{0}: {1}%".format(countries09[index], n*100))
#     index +=1
# print("Total: {0}%".format(np.sum(numbers09)*100))
# 
# 