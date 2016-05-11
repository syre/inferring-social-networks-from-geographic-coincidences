#!/usr/bin/env python3
import flask
from flask import Flask, request, \
     render_template, abort
import sys
import os
from multiprocessing.pool import ThreadPool
from urllib.parse import unquote_plus

sys.path.append(os.path.join("..", "tools"))
import DatabaseHelper
import GeoData
import main

app = Flask(__name__)
app.debug = True

tools_path = "../tools/"

database = DatabaseHelper.DatabaseHelper(tools_path)
run = main.Run()

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


def get_cooccurrences_async(useruuid1, useruuid2=None, points_w_distances=[]):
        print("App: Henter Geo-data data...")
        # tuple of args for foo, please note a "," at the end of the arguments
        async_result = pool.apply_async(g.get_geo_data_from_occurrences, (useruuid1, useruuid2, points_w_distances))
        return async_result.get()


def get_cooccurrences_async_all(country):
        print("App: Henter Geo-data data...")
        # tuple of args for foo, please note a "," at the end of the arguments
        async_result = pool.apply_async(g.get_geo_data_from_all_cooccurrences)
        return async_result.get()


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error=404), 404


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/cooccurrences")
def occurrences():
    useruuid1 = request.args.get("useruuid1")
    useruuid2 = request.args.get("useruuid2")

    user_list = database.get_users_with_most_updates()
    return render_template("cooccurrences_map.html", useruuid1=useruuid1, useruuid2=useruuid2, user_list=user_list)


@app.route("/distributions/cooccurrences")
def cooccurrences_distribution():
    # default users
    user_x = "8d325d9f-9341-4d00-a890-2adaf412e5ca"
    user_y = "420b91b6-5c78-4fe6-af7c-9795edd10c0e"
    useruuid = request.args.get("useruuid")
    if useruuid:
        user_x, user_y = useruuid.split(",")
    return render_template("distributions.html", feature="cooccurrences", user_x=user_x, user_y=user_y)


@app.route("/distributions/<feature>")
def distributions(feature):
    return render_template("distributions.html", feature=feature)


@app.route("/data/distributions/cooccurrences")
def data_distributions_cooccurrences():
    useruuid = request.args.get("useruuid")
    if useruuid:
        user_x, user_y = useruuid.split(",")
    data = database.get_distribution_cooccurrences(user_x, user_y)
    return flask.jsonify(results=data, user_x=user_x, user_y=user_y)


@app.route("/data/distributions/<feature>")
def data_distributions(feature):
    if feature in number_features:
        data = database.get_distributions_numbers(feature, num_bins=10, max_value=100000)
    elif feature in category_features:
        data = database.get_distributions_categories(feature, num_bins=10)
    else:
        abort(400)
    return flask.jsonify(results=data)


@app.route("/data/boxplot/<feature>")
def data_boxplot(feature):
    feature_values = request.args.get("values")
    if feature_values:
        feature_values = unquote_plus(feature_values)
    print("data_boxplot: feature_values = {0}".format(feature_values))
    data = []
    if feature == "country":
        if not feature_values:
            results, names = database.get_boxplot_duration("Denmark", for_all_countries=True)
            data.append({"results": results, "names": names, "id": "All countries"})
        else:
            feature_values = feature_values.split(",")
            for country in feature_values:
                results, names = database.get_boxplot_duration(country, for_all_countries=False)
                data.append({"results": results, "names": names, "id": country})
    elif feature == "velocity":
        if not feature_values:
            print("trfgg")
        else:
            feature_values = feature_values.split(",")
            for country in feature_values:
                results, names = database.get_velocity_for_users(country)
                data.append({"results": results, "names": names, 'id': country})
    return flask.jsonify(results=data)


@app.route("/boxplot/<feature>")
def boxplot(feature):
    all_countries = database.get_distinct_feature("name", "country")
    print(feature)
    if not feature:
        feature = all_countries[0]
        print(feature)
    return render_template("boxplot.html", feature=feature, all_countries=all_countries)


@app.route("/network")
def network():
    return render_template("network.html")


@app.route("/data/network")
def data_network():
    data = database.get_all_cooccurrences_as_network()
    return flask.jsonify(results=data)


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
    useruuid1 = request.args.get("useruuid1")
    useruuid2 = request.args.get("useruuid2")
    if useruuid1 == "None":
        useruuid1 = None
    if useruuid2 == "None":
        useruuid2 = None
    all_filters = [item for sublist in run.filter_places_dict.values() for item in sublist]
    if useruuid1 and useruuid2:
        cooccurrences = get_cooccurrences_async(useruuid1, useruuid2, points_w_distances=all_filters)
    elif useruuid1:
        cooccurrences = get_cooccurrences_async(useruuid1, points_w_distances=all_filters)
    else:
        print("getting from japan")
        cooccurrences = get_cooccurrences_async_all("Japan")
    return flask.jsonify(cooccurrences)


if __name__ == "__main__":
    app.run()
