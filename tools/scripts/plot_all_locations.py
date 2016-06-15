#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from DatabaseHelper import DatabaseHelper
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

boundaries = {"Sweden":[7, 20, 55, 65], "Japan":[125, 150, 30, 45]}
gridsizes = {"Sweden":250, "Japan":1000}
country = "Japan"

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
m.fillcontinents(color='#555555', zorder=0)
m.drawmapboundary(zorder=0)
x, y = m([l[0] for l in locations], [l[1] for l in locations])

m.hexbin(np.array(x), np.array(y), gridsize=gridsizes[country], bins="log", lw=0.2, alpha=1., mincnt=1, edgecolor="none", cmap=plt.get_cmap("Blues"))
if country == "Sweden":
    mob_coords = (13.2236973, 55.7172633)
    ericsson_coords = (11.9387722, 57.70585)
    sony_coords = (17.9535443, 59.4066036)
    x, y = m(mob_coords[0], mob_coords[1])
    x2, y2 = m(mob_coords[0]-0.7, mob_coords[1]-0.7)
    plt.annotate("Sony Mobile", xy=(x, y), xytext=(x2, y2), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="fancy", color='k'))
    x, y = m(ericsson_coords[0], ericsson_coords[1])
    x2, y2 = m(ericsson_coords[0]-1.8, ericsson_coords[1]+0.3)
    plt.annotate("Ericsson", xy=(x, y), xytext=(x2, y2), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="fancy", color='k'))
    x, y = m(sony_coords[0], sony_coords[1])
    x2, y2 = m(sony_coords[0]+0.3, sony_coords[1]-0.6)
    plt.annotate("Sony", xy=(x, y), xytext=(x2, y2), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="fancy", color='k'))
elif country == "Japan":
    sony_coords = (139.7609281, 35.6721699)
    x, y = m(sony_coords[0], sony_coords[1])
    x2, y2 = m(sony_coords[0]+1.5, sony_coords[1]-1.5)
    plt.annotate("Sony", xy=(x, y), xytext=(x2, y2), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="fancy", color='k'))

plt.tight_layout()
plt.show()
