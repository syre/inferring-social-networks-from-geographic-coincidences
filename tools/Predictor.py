#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
import datetime
import json
database = DatabaseHelper.DatabaseHelper()

class Predictor():
    def __init__(self, timebin_size_in_minutes):
        self.min_datetime = None
        self.max_datetime = None
        self.timebin_size = timebin_size_in_minutes
    
    def generate_dataset(self, country, cell_size):
        self.min_datetime = database.get_min_start_time_for_country(country)
        self.max_datetime = database.get_max_start_time_for_country(country)
        lst = []
        all_users_in_country = database.get_users_in_country(country)
        i_total = len(all_users_in_country)
        print_thresshold = math.floor(i_total/25)
        print("Number of users: {}".format(i_total))
        i = 1
        for user in all_users_in_country:
            if i%print_thresshold==0:
                print("users calculated: {}".format(i))
            cooccurrences = database.find_cooccurrences(user, cell_size, self.timebin_size)
            for occurrence in cooccurrences:
                time_bins = self.map_time_to_timebins(occurrence[1], occurrence[2])
                lng_lat = json.loads(occurrence[3])["coordinates"]
                spatial_bin = self.calculate_spatial_bin(lng_lat[0], lng_lat[1])
                for time_bin in time_bins:
                    lst.append([time_bin, spatial_bin, 1])
            i+=1
        print(len(lst))


    def predict(self):
        pass

    def map_time_to_timebins(self, start_time, end_time):
        duration = end_time-start_time
        duration = duration.total_seconds()/60.0 #in minutes
        start_diff = (start_time-self.min_datetime).total_seconds()/60.0
        start_bin = math.floor(start_diff/self.timebin_size) #tag højde for 0??
        end_bin = math.ceil((duration/self.timebin_size))
        return range(start_bin, start_bin+end_bin+1)
            
    def calculate_spatial_bin(self, lng, lat, resolution_decimals=3):
        GRID_MAX_LAT = 180 * pow(10,resolution_decimals)

        lat += 90.0
        lng += 180.0

        lat = math.trunc(lat*pow(10,resolution_decimals))
        lng = math.trunc(lng*pow(10,resolution_decimals))

        return (GRID_MAX_LAT * lat) + lng


# [user1, user2, timebin, geo, ja/nej]
# [0, 0, 1, 1, 0]

# [0,1, 2 ......]
# 0...2000