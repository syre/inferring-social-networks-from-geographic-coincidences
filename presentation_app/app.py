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
try:
    pool = ThreadPool(processes=1)
except Exception:
    print("Kunne ikke starte tråd")

def get_geodata_async(country, date):
        print("App: Henter Geo-data data...")
        # tuple of args for foo, please note a "," at the end of the arguments
        async_result = pool.apply_async(g.get_and_generate, (country,date))
        return async_result.get()


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/distributions/<feature>")
def distributions(feature):
    return render_template("distributions.html", feature=feature)

@app.route("/data/distributions/<feature>")
def data_distributions(feature):
    data = database.get_distributions(feature,num_bins=10)
    return flask.jsonify(results=data)

@app.route("/data/geojson")
def data_geojson():
    requested_date = request.args.get("date")
    print(requested_date)
    print("data_geojson aktiveret")
    gjson_data = get_geodata_async("Japan", requested_date)
    print(g.check_validity(gjson_data))
    print("Geo-data hentet!!")
    return flask.jsonify(gjson_data)

if __name__ == "__main__":
    app.run()