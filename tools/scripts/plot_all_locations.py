#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from DatabaseHelper import DatabaseHelper
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

db = DatabaseHelper()
locations = db.get_locations_by_country_only("Sweden")
locations = [(json.loads(l[3])["coordinates"][0], json.loads(l[3])["coordinates"][1]) for l in locations]

plt.figure(figsize=(20, 50))
m = Basemap(projection="merc", llcrnrlon=0, urcrnrlon=30,\
            llcrnrlat=55, urcrnrlat=65, resolution='h')
m.drawcoastlines(zorder = 0)
m.drawcountries(zorder = 0)
m.fillcontinents(color = 'lightgray', zorder = 0)
m.drawmapboundary(zorder = 0)
x, y = m([l[0] for l in locations], [l[1] for l in locations])
m.plot(x, y, "bo", markersize=5, alpha=0.1, zorder=1)
plt.show()
