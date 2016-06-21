#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import FileLoader
from collections import Counter
import seaborn as sns
import DatabaseHelper
import matplotlib.pyplot as plt

d = DatabaseHelper.DatabaseHelper()
distinct_users = d.get_distinct_feature("useruuid", "location")

colors = ['#3498db', '#e74c3c']
sns.set_palette(colors)
file_loader = FileLoader.FileLoader()
user_info_dict = file_loader.generate_demographics_from_csv()

print("users not in user_info_dict: {}".format(str(sorted(list(set(distinct_users)-set(user_info_dict.keys()))))))
print("------------------------------------------")
# ('1980-12-01', 344), ('1981-01-01', 85), ('1976-05-16', 31), ('1990-01-01', 20)
#user_info_dict = {k: v for k, v in user_info_dict.items() if v["birthdate"] != "1981-01-01" and v["birthdate"] != "1980-12-01"}
ages = [value["age"] for value in user_info_dict.values()]
birthdates = [value["birthdate"] for value in user_info_dict.values()]
#print([u for u in user_info_dict.values() if u["age"] < 16])
#print(len([u for u in user_info_dict.values() if u["birthdate"].endswith("-01")]), len(user_info_dict.values()))
print(Counter(birthdates).most_common(20))
ax = sns.distplot(ages, kde=False)
ax.set(xlabel="Age", title="Distribution of user ages", ylabel="Frequency")
ax.set_xlim(0)
plt.gca().title.set_fontsize(20)
[item.set_fontsize(14) for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()]
[item.set_fontsize(14) for item in [plt.gca().yaxis.label, plt.gca().xaxis.label]]
#sns.boxplot(y=ages)
sns.plt.show()