#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
import datetime
import json
import numpy as np
import scipy
import scipy.sparse
import sklearn.metrics.pairwise

class Predictor():
    def __init__(self, timebin_size_in_minutes, spatial_resolution_decimals = 3):
        self.database = DatabaseHelper.DatabaseHelper()
        self.min_datetime = None
        self.max_datetime = None
        self.timebin_size = timebin_size_in_minutes
        self.spatial_resolution_decimals = spatial_resolution_decimals
        self.GRID_MAX_LNG = 180 * pow(10, spatial_resolution_decimals)
        self.GRID_MAX_LAT = 90 * pow(10, spatial_resolution_decimals)
    
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
            
    def calculate_spatial_bin(self, lng, lat):
        lat += 90.0
        lng += 180.0

        lat = math.trunc(lat*pow(10,self.spatial_resolution_decimals))
        lng = math.trunc(lng*pow(10,self.spatial_resolution_decimals))

        return (self.GRID_MAX_LAT * lat) + lng

    def calculate_corr(self, user1, user2, resolution_decimals=3):
        """
        Due to limited number of possible locations, the activity of each user can be de-
        scribed as a set of binary vectors of presence at each location. Each dyad of users
        can then be assigned a similarity score by calculating the correlation between
        their activity vectors for each building and then summing all the statistically
        significant correlations. 

        Feature ID: corr

        """
        user1_locations = self.database.get_locations_for_user(user1)
        user2_locations = self.database.get_locations_for_user(user2)

        

        pass
        
    def calculate_location_vector(self, locations):
        """
        calculate vector for a given user, where each index is a possible location in the world
        and the entry on that index is 1 if the user has been there or 0 otherwise
        """
        user_vector = {}
        for location in locations:
            lng = location[3]
            lat = location[4]
            user_vector[self.calculate_spatial_bin(lng, lat)] = 1
        return user_vector

    
    def calculate_arr_leav(self, user1, user2):
        """
        We propose that if two persons arrive at a location at the same time and/or
        leave the location synchronously it yields a stronger signal than if two people
        are in the same location, but their arrival and leaving are not synchronized. The
        value is weighted by the number of people who arrived and/or left the building
        in each particular time bin. Thus, timed arrival of many people in the beginning
        of the scheduled classes is not as strong a signal as synchronized arrival of a few
        persons an hour before the class begins. 

        Feature ID: arr_leav
        """
        pass
    
    def calculate_coocs_w(self, user1, user2):
        """
        While other researchers use entropy to weight the social impact of meetings, our
        data allows us to introduce a more precise measure. We use anonymous statistics
        to estimate the number of all people present in the building in each time bin. We
        assume the social importance of each co-occurrence to be inversely proportional
        to the number of people – if only a few persons are there in a location, it is more
        probable that there is a social bond between them compared to the situation
        when dozens of people are present. Feature ID: coocs_w, see Figure 4.8b.
        """
    
    def calculate_specificity(self, user1, user2):
        """
        The asymmetric specificity Sij defined as fraction of time person pi spends with
        person pj with respect to the total time spent on campus by person pj . As
        shown in Table 4.3, the fraction of social time with respect to total time is more
        indicative of being perceived as a friend than only the social time (ID: spec,
        Figure 4.8p).

        """
        pass

if __name__ == '__main__':
    p = Predictor(20.0)
    print(p.calculate_corr("175ceb15-5a9a-4042-9422-fcae763fe305", "2ddb668d-0c98-4258-844e-7e790ea65aba", 3))