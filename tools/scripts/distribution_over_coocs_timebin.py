#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader
import numpy as np
import seaborn as sns

file_loader = FileLoader()
database_helper = DatabaseHelper()

dataset_helper = DatasetHelper()

country = "Japan"
predictor = Predictor(country)
filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}


sept_min_datetime = "2015-09-01 00:00:00+00:00"
sept_min_time_bin = database_helper.calculate_time_bins(sept_min_datetime, sept_min_datetime)[0]
sept_max_datetime = "2015-09-30 23:59:59+00:00"
sept_max_time_bin = database_helper.calculate_time_bins(sept_max_datetime, sept_max_datetime)[0]
oct_min_datetime = "2015-10-01 00:00:00+00:00"
oct_min_time_bin = database_helper.calculate_time_bins(oct_min_datetime, oct_min_datetime)[0]
oct_max_datetime = "2015-10-31 23:59:59+00:00"
oct_max_time_bin = database_helper.calculate_time_bins(oct_max_datetime, oct_max_datetime)[0]
nov_min_datetime = "2015-11-01 00:00:00+00:00"
nov_min_time_bin = database_helper.calculate_time_bins(nov_min_datetime, nov_min_datetime)[0]
nov_max_datetime = "2015-11-30 23:59:59+00:00"
nov_max_time_bin = database_helper.calculate_time_bins(nov_max_datetime, nov_max_datetime)[0]

print("processing users, countries and locations as numpy matrix (train)")
users_train, countries_train, locations_train = database_helper.generate_numpy_matrix_from_database()
locations_train = predictor.filter_by_country(locations_train, countries_train)

print("processing users, countries and locations as numpy matrix (test)")
users_test, countries_test, locations_test = database_helper.generate_numpy_matrix_from_database(filter_places_dict[country])
locations_test = predictor.filter_by_country(locations_test, countries_test)

coocs_train = dataset_helper.generate_cooccurrences_array(locations_train)
coocs_test = dataset_helper.generate_cooccurrences_array(locations_test)

train_sep = coocs_train[coocs_train[:, 3] <= sept_max_time_bin]
train_sep = train_sep[train_sep[:, 3] >= sept_min_time_bin]
test_sep = coocs_test[coocs_test[:, 3] <= sept_max_time_bin]
test_sep = test_sep[test_sep[:, 3] >= sept_min_time_bin]

train_oct = coocs_train[coocs_train[:, 3] <= oct_max_time_bin]
train_oct = train_oct[train_oct[:, 3] >= oct_min_time_bin]
test_oct = coocs_test[coocs_test[:, 3] <= oct_max_time_bin]
test_oct = test_oct[test_oct[:, 3] >= oct_min_time_bin]

train_nov = coocs_train[coocs_train[:, 3] <= nov_max_time_bin]
train_nov = train_nov[train_nov[:, 3] >= nov_min_time_bin]
test_nov = coocs_test[coocs_test[:, 3] <= nov_max_time_bin]
test_nov = test_nov[test_nov[:, 3] >= nov_min_time_bin]


train_cooc_timebin_sep = train_sep[:, 3]
test_cooc_timebin_sep = test_sep[:, 3]
train_cooc_timebin_oct = train_oct[:, 3]
test_cooc_timebin_oct = test_oct[:, 3]
train_cooc_timebin_nov = train_nov[:, 3]
test_cooc_timebin_nov = test_nov[:, 3]

train_cooc_timebin_sep.sort()
test_cooc_timebin_sep.sort()
train_cooc_timebin_oct.sort()
test_cooc_timebin_oct.sort()
train_cooc_timebin_nov.sort()
test_cooc_timebin_nov.sort()

train_sep_labels = list(set(train_cooc_timebin_sep))
test_sep_labels = list(set(test_cooc_timebin_sep))
train_oct_labels = list(set(train_cooc_timebin_oct))
test_oct_labels = list(set(test_cooc_timebin_oct))
train_nov_labels = list(set(train_cooc_timebin_nov))
test_nov_labels = list(set(test_cooc_timebin_nov))


train_sep_labels.sort()
test_sep_labels.sort()
train_oct_labels.sort()
test_oct_labels.sort()
train_nov_labels.sort()
test_nov_labels.sort()

data = [(train_cooc_timebin_sep, train_sep_labels, "Train - september"),
        (test_cooc_timebin_sep, test_sep_labels, "Test - september"),
        (train_cooc_timebin_oct, train_oct_labels, "Train - october"),
        (test_cooc_timebin_oct, test_oct_labels, "Test - october"),
        (train_cooc_timebin_nov, train_nov_labels, "Train - november"),
        (test_cooc_timebin_nov, test_nov_labels, "Test - november")]

for plot in range(6):
    print(plot)
    sns.plt.subplot(3, 2, plot+1)  
    test = {"test2": data[plot][0]}  
    ax = sns.countplot(x="test2", data=test, label='big')
    #sns.set(font_scale=2.5)
    ax.set_xticklabels(data[plot][1], rotation=90)
    ax.set_xlabel("Timebins", fontsize=35)
    ax.set_ylabel("Frequency", fontsize=35)
    #ax.set(ylabel="Count")
    ax.set(title=data[plot][2])
    [label.set_visible(False) for label in ax.xaxis.get_ticklabels()]

    for label in ax.xaxis.get_ticklabels()[::10]:
        label.set_visible(True)
sns.plt.show()