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

number_features = ["altitude", "accuracy"]
category_features = ["country", "region", "area", "place"]

g = GeoData.GeoData(tools_path)
try:
    pool = ThreadPool(processes=1)
except Exception:
    print("Kunne ikke starte tråd")

def get_geodata_async(country, start_date, end_date):
        print("App: Henter Geo-data data...")
        # tuple of args for foo, please note a "," at the end of the arguments
        async_result = pool.apply_async(g.get_and_generate, (country, start_date, end_date))
        return async_result.get()

def get_cooccurrences_async(useruuid, cell_size, time_threshold):
        print("App: Henter Geo-data data...")
        # tuple of args for foo, please note a "," at the end of the arguments
        async_result = pool.apply_async(g.get_geo_data_from_occurrences, (useruuid, cell_size, time_threshold))
        return async_result.get()

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error=404), 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/cooccurrences")
def occurrences():
    useruuid = request.args.get("useruuid")
    if not useruuid:
        # get default user japan
        useruuid = "e21901af-70ba-402c-9e98-92fd6e0656f6"
    user_list = database.get_users_with_most_updates()
    return render_template("cooccurrences_map.html", useruuid=useruuid, user_list=user_list)

@app.route("/distributions/<feature>")
def distributions(feature):
    print("distributions render_template")
    return render_template("distributions.html", feature=feature)

@app.route("/data/distributions/<feature>")
def data_distributions(feature):
    if feature in number_features:
        data, layout = database.get_distributions_numbers(feature,num_bins=10)
    elif feature in category_features:
        data, layout = database.get_distributions_text(feature,num_bins=10)
    else:
        abort(400)
    print("distributions")
    return flask.jsonify(results=data, x_axis=layout['x_axis'], y_axis=layout['y_axis'])

@app.route("/data/multi_boxplot/<feature>")
def data_multi_boxplot(feature):
    # /country?values=Japan,Sweden,Denmark
    #
    #
    print("data multi_boxplot - feature = {0}".format(feature))
    feature_values = request.args.get("values")
    print(feature_values)
    
    #countries = ["Japan", "Sweden", "Denmark", "Finland", "Germany"]
    
    data = []
    if feature == "country":
        if not feature_values:
            results, names = database.get_boxplot_duration("Denmark", for_all_countries=True)
            data.append({"results":results, "names":names, "id":"All countries"})
        else:
            feature_values = feature_values.split(",")
            for country in feature_values:
                results, names = database.get_boxplot_duration(country, for_all_countries=False)
                data.append({"results":results, "names":names, "id":country})
    return flask.jsonify(results=data)

@app.route("/boxplot/<feature>")
def boxplot(feature):
    all_countries = database.get_feature_sql_as_list("SELECT name from country order by name;")
    print(feature)
    if not feature:
        feature = all_countries[0][0]
        print(feature)
    return render_template("boxplot.html", feature=feature, all_countries=all_countries[0])

@app.route("/data/boxplot/<feature>")
def data_boxplot(feature):
    data, names = database.get_boxplot_duration("Japan", for_all_countries=True)
    return flask.jsonify(results=data, names=names, feature='country')


@app.errorhandler(400)
def page_not_found(e):
    return render_template('error.html'), 400

@app.route("/data/geojson")
def data_geojson():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    gjson_data = get_geodata_async("Japan", start_date, end_date)

    return flask.jsonify(gjson_data)

@app.route("/data/cooccurrences")
def data_cooccurrences():
    useruuid = request.args.get("useruuid")
    cooccurrences = get_cooccurrences_async(useruuid, 0.001, 60*24)

    return flask.jsonify(cooccurrences)


if __name__ == "__main__":
    app.run()