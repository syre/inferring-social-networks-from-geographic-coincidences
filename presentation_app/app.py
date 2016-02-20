#!/usr/bin/env python3
import flask
import json
from flask import Flask, request, flash, url_for, redirect, \
     render_template, abort
import sys
import os
from threading import Thread
from multiprocessing.pool import ThreadPool

sys.path.append(os.path.join("..","tools"))
import DatabaseHelper
import GeoData

app = Flask(__name__)
app.debug = True

tools_path = "../tools/"
database = DatabaseHelper.DatabaseHelper(tools_path)
g = GeoData.GeoData(tools_path)

#print("App: Henter Geo-data data...")
try:
    pool = ThreadPool(processes=1)
    print("App: Henter Geo-data data...")
    # tuple of args for foo, please note a "," at the end of the arguments
    async_result = pool.apply_async(g.get_and_generate, ('Japan',))
    #print("App: Geo-data hentet!")
    #geo_data = Thread.(target=g.get_and_generate, args=("Japan"))
except Exception:
    print("Kunne ikke starte tråd")




@app.route('/')
def index():
    return render_template('index.html')

@app.route("/distributions/<feature>")
def distributions(feature):
	return render_template("distributions.html", feature=feature)

@app.route("/data/distributions/<feature>")
def data_distributions(feature):
    data = database.get_distributions(feature,num_bins=10) #database_helper.get_distributions(feature,num_bins=10)
    return flask.jsonify(results=data)

@app.route("/data/geojson")
def data_geojson():
    print("data_geojson aktiveret")
    gjson_data = async_result.get()
    print(g.check_validity(gjson_data))
    print("Geo-data hentet!!")
    #print(g.check_validity(gjson_data))
	#with open("geodata.geojson") as f:
    #data = json.load(gjson_data) 
    for x in gjson_data["features"]:
        print(x['properties']['id'])
    #print(gjson_data["features"][0]['properties']['id'])  
    return flask.jsonify(gjson_data)

if __name__ == "__main__":
    app.run()