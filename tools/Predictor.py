#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
database = DatabaseHelper.DatabaseHelper()

class Predictor():
	def __init__(self, min_datetime, max_datetime, timebin_size):
        self.min_datetime = min_datetime
        self.max_datetime = max_datetime
        self.timebin_size = timebin_size
		pass
	
	def generate_dataset(self, country, cell_size, time_threshold_in_minutes):
        all_users_in_country = database.get_users_in_country("Japan")
        for user in all_users_in_country:
            cooccerrences = database.find_cooccurrences(user, cell_size, time_threshold_in_minutes)
            for occurrence in cooccerrences:
                pass
            """useruuid, start_time, end_time, location AS geom"""
		pass



    def predict(self):
        pass

    def map_time_to_timebin(self, start_time, end_time):
            
	def calculate_spatial_bin(lng, lat, resolution_decimals=3):
		GRID_MAX_LAT = 180 * pow(10,resolution_decimals)

		lat += 90
		lng += 180

		lat = math.trunc(lat*pow(10,resolution_decimals))
		lng = math.trunc(lng*pow(10,resolution_decimals))

		return (GRID_MAX_LAT * lat) + lng


[user1, user2, timebin, geo, ja/nej]
[0, 0, 1, 1, 0]

[0,1, 2 ......]
0...2000