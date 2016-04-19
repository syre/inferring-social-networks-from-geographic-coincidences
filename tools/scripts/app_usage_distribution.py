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
d = DatabaseHelper.DatabaseHelper()
country_users = d.get_users_in_country("Japan")


sns.set(color_codes=True)
sns.set(style="white", palette="muted")
file_loader = FileLoader.FileLoader()

rows = []
file_loader.generate_app_data_from_json(lambda r: rows.append(r))
feature = "com.android.incallui"
rows = [r for r in rows if r["package_name"] == feature and r["useruuid"] in country_users]
start_end_pairs = [(parser.parse(r["start_time"]), parser.parse(r["end_time"])) for r in rows]
bin_size = 5
values = []
for start, end in start_end_pairs:
    for x in range(int((start.hour*60+start.minute)/bin_size), math.ceil((end.hour*60+end.minute)/bin_size)):
        values.append(x)

ax = sns.distplot(values, bins=(24*60+60)/bin_size)
#ax.set(xticks=times_list)
ax.set(xlabel="clock")
#ax.set_xlim(0, 23)
sns.plt.show()