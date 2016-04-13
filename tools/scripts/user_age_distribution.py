#!/usr/bin/env python3
import seaborn as sns
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import FileLoader
from collections import Counter

sns.set(color_codes=True)
sns.set(style="white", palette="muted")
file_loader = FileLoader.FileLoader()
user_info_dict = file_loader.generate_demographics_from_csv()
# 1980-12-01', 165), ('1981-01-01', 47), ('1990-01-01', 13)
#user_info_dict = {k: v for k, v in user_info_dict.items() if v["birthdate"] != "1990-01-01" and v["birthdate"] != "1981-01-01" and v["birthdate"] != "1980-12-01"}
ages = [value["age"] for value in user_info_dict.values()]
birthdates = [value["birthdate"] for value in user_info_dict.values()]
print(Counter(birthdates).most_common(20))
sns.distplot(ages)
sns.plt.show()