#!/usr/bin/env python3
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DatabaseHelper import DatabaseHelper
from collections import defaultdict
db = DatabaseHelper()

locations = db.get_locations_by_country_only("Japan")

useruuids = set([l[0] for l in locations])

user_location_dict = defaultdict(list)
for location in locations:
	user_location_dict[location[0]].append((location[1], location[2]))

avg_times_between = []
for user, times in user_location_dict.items():
	avg_time = 0
	times.sort(key= lambda d: d[0])
	for x in range(len(times)-1):
		avg_time += ((times[x+1][1]-times[x][0]).seconds/60)
	avg_time /= len(times)
	avg_times_between.append(avg_time)
print(sum(avg_times_between)/len(avg_times_between))
