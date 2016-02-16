#!/usr/bin/env python3
import flask
import json
from flask import Flask, request, flash, url_for, redirect, \
     render_template, abort
import sys
import os

sys.path.append(os.path.join("..","tools"))
import database_helper

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/distributions/<feature>")
def distributions(feature):
	return render_template("distributions.html", feature=feature)

@app.route("/data/distributions/<feature>")
def data_distributions(feature):
    data = database_helper.get_distributions(feature,num_bins=10)
    return flask.jsonify(results=data)

@app.route("/data/geojson")
def data_geojson():
	with open("geodata.geojson") as f:
		data = json.load(f)
	return flask.jsonify(data)

if __name__ == "__main__":
    app.run()