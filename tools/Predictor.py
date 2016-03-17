#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
import datetime
import json
from datetime import datetime
from pytz import timezone
import numpy as np
import scipy
from scipy import sparse, stats
import sklearn

class Predictor():
    def __init__(self, 
                 timebin_size_in_minutes,
                 from_date=datetime.strptime("2015-09-01", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo")),
                 to_date=datetime.strptime("2015-11-30", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo")),
                 grid_boundaries_tuple=(-180, 180, -90, 90), spatial_resolution_decimals = 3):
        """
            Constructor

            Args:
                timebin_size_in_minutes: size of timebin in minutes - for example 60 to get hourly timebin
                spatial_resolution_decimals: spatial resolution - used for spatial binning, for example 3 to get lng and lat down to 0.001 precision
                grid_boundaries_tuple: boundaries of map - used for spatial binning,
                                       Japan for example is within the range of lng: 120 - 150 and lat: 20-45, thus (120, 150, 20, 45)
                from_date: start-date used for timebin range
                to_date: end-date used for timebin range
        """
        self.database = DatabaseHelper.DatabaseHelper()
        self.min_datetime = from_date
        self.max_datetime = to_date
        self.timebin_size = timebin_size_in_minutes
        self.spatial_resolution_decimals = spatial_resolution_decimals

        self.GRID_MIN_LNG = (grid_boundaries_tuple[0] + 180) * pow(10, spatial_resolution_decimals)
        self.GRID_MAX_LNG = (grid_boundaries_tuple[1] + 180) * pow(10, spatial_resolution_decimals)
        self.GRID_MIN_LAT = (grid_boundaries_tuple[2] + 90) * pow(10, spatial_resolution_decimals)
        self.GRID_MAX_LAT = (grid_boundaries_tuple[3] + 90) * pow(10, spatial_resolution_decimals)
        print("GRID_MIN_LNG = {}\nGRID_MAX_LNG = {}\nGRID_MIN_LAT = {}\nGRID_MAX_LAT = {}\n------------------".
            format(self.GRID_MIN_LNG, self.GRID_MAX_LNG, self.GRID_MIN_LAT, self.GRID_MAX_LAT))
    
    def generate_dataset(self, country, cell_size):
        lst = []
        all_users_in_country = self.database.get_users_in_country(country)
        i_total = len(all_users_in_country)
        print_thresshold = math.floor(i_total/25)
        print("Number of users: {}".format(i_total))
        i = 1
        for user in all_users_in_country:
            if i%print_thresshold==0:
                print("users calculated: {}".format(i))
            cooccurrences = self.database.find_cooccurrences(user, cell_size, self.timebin_size)
            for occurrence in cooccurrences:
                time_bins = self.map_time_to_timebins(occurrence[1], occurrence[2])
                lng_lat = json.loads(occurrence[3])["coordinates"]
                print("lat: {}\nlng: {}".format(lng_lat[1],lng_lat[0]))
                spatial_bin = self.calculate_spatial_bin(lng_lat[0], lng_lat[1])
                print("spatial_bin = {}".format(spatial_bin))
                break
                for time_bin in time_bins:
                    lst.append([time_bin, spatial_bin, 1])
            i+=1
            break
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
        return (abs(self.GRID_MAX_LAT - self.GRID_MIN_LAT) * (lat-self.GRID_MIN_LAT)) + (lng-self.GRID_MIN_LNG)



    
    def jaccard_index(self, X, Y):
        print(X.nonzero())
        print(Y.nonzero())
    
    def find_users_in_cooccurrence(self, lng, lat, time_bin):
        """
        Find all users who's been in a given cooccurrence
            Arguments:
                lat {float} -- latitude
                lng {float} -- longitude
                time_bin {integer} -- time bin index
            Returns:
                list -- list of user_uuids
        
        """
        lat = math.trunc(lat*pow(10,self.spatial_resolution_decimals))
        lng = math.trunc(lng*pow(10,self.spatial_resolution_decimals))
        #time_bin = 10
        #        
        start_time = self.min_datetime+(datetime.timedelta(minutes=self.timebin_size_in_minutes)*time_bin)
        #end_time = start_time+datetime.timedelta(minutes=self.timebin_size_in_minutes)
        return self.database.find_cooccurrences_within_area(lat, lng, start_time, self.timebin_size_in_minutes, self.spatial_resolution_decimals)
        #raise NotImplementedError

    def calculate_corr(self, user1, user2):
        """
        Due to limited number of possible locations, the activity of each user can be de-
        scribed as a set of binary vectors of presence at each location. Each dyad of users
        can then be assigned a similarity score by calculating the correlation between
        their activity vectors for each building and then summing all the statistically
        significant correlations. 

        Feature ID: corr

        """
        correlation_sum = 0
        user1_locations = self.database.get_locations_for_user(user1)
        user2_locations = self.database.get_locations_for_user(user2)
        time_bin_range = self.map_time_to_timebins(self.min_datetime, self.max_datetime)
        array_size = abs(self.GRID_MAX_LAT-self.GRID_MIN_LAT)*abs(self.GRID_MAX_LNG-self.GRID_MIN_LNG)

        for bin in time_bin_range:
            user1_vector = np.zeros(array_size, dtype=bool)
            for location in user1_locations:
                start_time = location[1]
                end_time = location[2]
                lng = location[3]
                lat = location[4]
                if bin in self.map_time_to_timebins(start_time, end_time):
                    user1_vector[self.calculate_spatial_bin(lng, lat)] = 1
                    break
            user2_vector = np.zeros(array_size, dtype=bool)
            for location in user2_locations:
                start_time = location[1]
                end_time = location[2]
                lng = location[3]
                lat = location[4]
                if bin in self.map_time_to_timebins(start_time, end_time):
                    print(lng, lat)
                    user2_vector[self.calculate_spatial_bin(lng, lat)] = 1
                    break

            correlation = stats.pearsonr(user1_vector, user2_vector)
            # if correlation is siginificant p < 0.05 add to correlation sum
            if correlation[1] < 0.05:
                correlation_sum += correlation[0]
        return correlation_sum

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
        cell_size = pow(10, -self.spatial_resolution_decimals)
        cooccurrences = self.database.find_cooccurrences(user1, cell_size, self.timebin_size, useruuid2=user2)
        arr_leav_values = []
        for cooc in cooccurrences:
            lng_lat = json.loads(cooc[3])
            start_time = cooc[1]
            end_time = cooc[2]
            lng = lng_lat["coordinates"][0]
            lat = lng_lat["coordinates"][1]
            arr_leav_value = 0
            spatial_bin = self.calculate_spatial_bin(lng, lat)
            time_bins = self.map_time_to_timebins(start_time, end_time)
            # check if one of the users are in the previous timebin but not both
            previous_list = self.find_users_in_cooccurrence(spatial_bin,time_bins[0]-1)
            current_list = self.find_users_in_cooccurrence(spatial_bin, time_bins)
            next_list = self.find_users_in_cooccurrence(spatial_bin, time_bins[-1]+1)
            
            number_of_new_arrivals = len(set(current_list)-set(previous_list))
            number_of_leavers = len(set(next_list)-set(current_list))


            if (user1 in previous_list and user2 not in previous_list) or (user1 not in previous_list and user2 in previous_list):
                # non-synchronously arrival
                arr_leav_value += 0
            else:
                # synchronous arrival
                arr_leav_value += 1*(1/(number_of_new_arrivals))

            if (user1 in next_list and user2 not in next_list) or (user1 not in next_list and user2 in next_list):
                # non-synchronously leaving
                arr_leav_value += 0
            else:
                # synchronous leaving
                arr_leav_value += 1*(1/(number_of_leavers))
            
            arr_leav_values.append(arr_leav_value)
        return sum(arr_leav_values)/len(cooccurrences)


    def calculate_coocs_w(self, user1, user2):
        """
        While other researchers use entropy to weight the social impact of meetings, our
        data allows us to introduce a more precise measure. We use anonymous statistics
        to estimate the number of all people present in the building in each time bin. We
        assume the social importance of each co-occurrence to be inversely proportional
        to the number of people – if only a few persons are there in a location, it is more
        probable that there is a social bond between them compared to the situation
        when dozens of people are present.

        Calculates all values of coocs_w for cooccurrences and returns the mean of them 

        Feature ID: coocs_w
        """
        cell_size = pow(10, -self.spatial_resolution_decimals)
        cooccurrences = self.database.find_cooccurrences(user1, cell_size, self.timebin_size, useruuid2=user2)
        coocs_w_values = []
        for cooc in cooccurrences:
            lng_lat = json.loads(cooc[3])
            start_time = cooc[1]
            end_time = cooc[2]
            lng = lng_lat["coordinates"][0]
            lat = lng_lat["coordinates"][1]

            coocs_w_value = 0

            spatial_bin = self.calculate_spatial_bin(lng, lat)
            time_bins = self.map_time_to_timebins(start_time, end_time)

            for time_bin in time_bins:
                users = self.find_users_in_cooccurrence(spatial_bin, time_bin)
                num_users = len(users)
                coocs_w_value += num_users
            
                if num_users < 2:
                    raise Exception("no users for cooccurrence")
            coocs_w_value /= len(time_bins)

            # 2 users is ideal thus returning highest value 1, else return lesser value proportional to amount of users
            coocs_w_values.append(1/(coocs_w_value-1))

        return sum(coocs_w_values)/len(cooccurrences)
    
    def calculate_specificity(self, user1, user2):
        """
        The asymmetric specificity Sij defined as fraction of time person pi spends with
        person pj with respect to the total time spent on campus by person pj . As
        shown in Table 4.3, the fraction of social time with respect to total time is more
        indicative of being perceived as a friend than only the social time.

        Feature ID: spec

        """
        pass

if __name__ == '__main__':
    JAPAN_TUPLE = (120, 150, 20, 45)
    decimals = 2
    p = Predictor(60, grid_boundaries_tuple=JAPAN_TUPLE, spatial_resolution_decimals=decimals)
    #p.generate_dataset("Japan", 0.001)
    print(p.calculate_corr("492f0a67-9a2c-40b8-8f0a-730db06abf65", "4bd3f3b1-791f-44be-8c52-0fd2195c4e62"))
    #print(p.calculate_coocs_w("492f0a67-9a2c-40b8-8f0a-730db06abf65", "4bd3f3b1-791f-44be-8c52-0fd2195c4e62"))
    #print(p.calculate_arr_leav("492f0a67-9a2c-40b8-8f0a-730db06abf65", "4bd3f3b1-791f-44be-8c52-0fd2195c4e62"))