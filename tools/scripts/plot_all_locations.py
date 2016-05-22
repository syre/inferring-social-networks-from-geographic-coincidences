#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from DatabaseHelper import DatabaseHelper
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

boundaries = {"Sweden":[5, 30, 55, 70], "Japan":[125, 150, 30, 45]}
gridsizes = {"Sweden":125, "Japan":1000}
country = "Sweden"

db = DatabaseHelper()
locations = db.get_locations_by_country_only(country)
locations = [(json.loads(l[3])["coordinates"][0], json.loads(l[3])["coordinates"][1]) for l in locations]


llcnrlon = boundaries[country][0]
urcrnrlon = boundaries[country][1]
llcrnrlat = boundaries[country][2]
urcrnrlat = boundaries[country][3]

plt.figure(figsize=(20, 50))
m = Basemap(projection="merc", llcrnrlon=llcnrlon, urcrnrlon=urcrnrlon,
            llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, resolution='h')
m.drawcoastlines(zorder=0)
m.drawcountries(zorder=0)
m.fillcontinents(color='lightgray', zorder=0)
m.drawmapboundary(zorder=0)
x, y = m([l[0] for l in locations], [l[1] for l in locations])
#m.plot(x, y, "bo", markersize=5, alpha=0.1, zorder=1)
m.hexbin(np.array(x), np.array(y), gridsize=gridsizes[country], bins="log", lw=0.2, alpha=1., mincnt=1, edgecolor="none", cmap=plt.get_cmap("Blues"))
m.readshapefile("shapefiles/Distrikt_v1", "Distrikt_v1")
plt.tight_layout()
plt.show()
