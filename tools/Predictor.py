#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
database = DatabaseHelper.DatabaseHelper()

class Predictor():
	def __init__(self):
		pass
	
	def generate_dataset(self):
		pass

	def calculate_spatial_bin(lng, lat, resolution_decimals=3):
		GRID_MAX_LAT = 180 * pow(10,resolution_decimals)

		lat += 90
		lng += 180

		lat = math.trunc(lat*pow(10,resolution_decimals))
		lng = math.trunc(lng*pow(10,resolution_decimals))

		return (GRID_MAX_LAT * lat) + lng


