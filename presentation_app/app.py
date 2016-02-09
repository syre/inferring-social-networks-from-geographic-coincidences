#!/usr/bin/env python3
import flask
import json
from flask import Flask, request, flash, url_for, redirect, \
     render_template, abort

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/data/geojson")
def data_geojson():
	with open("geodata.geojson") as f:
		data = json.load(f)
	return flask.jsonify(data)

if __name__ == "__main__":
    app.run()