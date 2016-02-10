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

@app.route("/distributions")
def distributions():
	return render_template("distributions.html")

@app.route("/data/distributions")
def data_distributions():
	with open("japan_data09.json") as f:
		data = json.load(f)

	return flask.jsonify(results=[
    {'Date': '01-03-2013', 'Views': 'a', 'Owner':'Alpha','Rating':'****'},
    {'Date': '05-03-2013', 'Views': 'b', 'Owner':'Beta','Rating':'****'},
    {'Date': '09-03-2013', 'Views': 'c', 'Owner':'Gamma','Rating':'**'},
    {'Date': '13-03-2013', 'Views': 'd', 'Owner':'Beta','Rating':'****'},
    {'Date': '01-04-2013', 'Views': 'a', 'Owner':'Theta','Rating':'****'},
    {'Date': '05-04-2013', 'Views': 'b', 'Owner':'Beta','Rating':'***'},
    {'Date': '09-04-2013', 'Views': 'c', 'Owner':'Theta','Rating':'**'},
    {'Date': '13-04-2013', 'Views': 'd', 'Owner':'Beta','Rating':'*'},
])

@app.route("/data/geojson")
def data_geojson():
	with open("geodata.geojson") as f:
		data = json.load(f)
	return flask.jsonify(data)

if __name__ == "__main__":
    app.run()