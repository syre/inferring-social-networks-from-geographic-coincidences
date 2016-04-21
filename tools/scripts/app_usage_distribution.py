#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import FileLoader
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import DatabaseHelper
from dateutil import parser
import math
from collections import defaultdict
import pprint
import collections
d = DatabaseHelper.DatabaseHelper()
from dateutil.relativedelta import relativedelta
import numpy as np

country_users = d.get_users_in_country("Japan")


sns.set(color_codes=True)
sns.set(style="white", palette="muted")
file_loader = FileLoader.FileLoader()

rows = []
file_loader.generate_app_data_from_json(lambda r: rows.append(r))
feature = "com.android.incallui"
rows = [r for r in rows if r["package_name"] == feature and r["useruuid"] in country_users]
start_end_pairs = [(parser.parse(r["start_time"]), parser.parse(r["end_time"])) for r in rows]
min_start = min([start for start, _ in start_end_pairs])
max_end = max([end for _, end in start_end_pairs])
print(min_start)
print(min_start.strftime("%H:%M"))
print(max_end)
bin_size = 5
values = []
time_values = defaultdict(dict)
for start, end in start_end_pairs:
    for x in range(int((start.hour*60+start.minute)/bin_size), math.ceil((end.hour*60+end.minute)/bin_size)):
        values.append(x)
        if (int((start.hour*60+start.minute)/bin_size)) not in time_values:
            time_values[int((start.hour*60+start.minute)/bin_size)] = start.strftime("%H:%M")

times = []
for x in range(min(values), max(values)):
    hours = math.floor((bin_size*x)/60)
    temp_time = min_start + relativedelta(minutes=(bin_size*x))
    times.append(temp_time.strftime("%H:%M"))


print(len(times))
ax = sns.distplot(values, bins=(24*60+60)/bin_size)
ax.set_xticks(np.arange(len(times)))
ax.set_xticklabels(times, rotation=90)
ax.set(xlabel="time of day")
sns.plt.tick_params(labelsize=14)

[label.set_visible(False) for label in ax.xaxis.get_ticklabels()]

for label in ax.xaxis.get_ticklabels()[::8]:
    label.set_visible(True)
sns.plt.show()
