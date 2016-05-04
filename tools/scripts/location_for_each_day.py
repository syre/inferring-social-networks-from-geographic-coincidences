#!/usr/bin/env python3
"""
Finds users with locations for all days of the three months
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DatabaseHelper import DatabaseHelper

db = DatabaseHelper()


locations = db.get_locations_by_country_only("Japan")

users = set([l[0] for l in locations])
sept_dates = range(1, 31)
oct_dates = range(1, 32)
nov_dates = range(1, 31)

all_days_users = []
for u in users:
    location_dates = [(l[1].date().day, l[1].date().month) for l in locations if l[0] == u]
    if set([(x,9) for x in sept_dates]) != set([x for x in location_dates if x[1] == 9]):
        continue
    if set([(x,10) for x in oct_dates]) != set([x for x in location_dates if x[1] == 10]):
        continue
    if set([(x,11) for x in nov_dates]) != set([x for x in location_dates if x[1] == 11]):
        continue
    all_days_users.append(u)

print(all_days_users)
print(len(all_days_users))