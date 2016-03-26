#!/usr/bin/env python3
import pybrain
import DatabaseHelper
import math
import datetime
from datetime import datetime, timedelta
from pytz import timezone
import numpy as np
from scipy import stats
import sklearn
import timeit
from tqdm import tqdm
import pickle

class Predictor():
    def __init__(self, 
                 timebin_size_in_minutes,
                 from_date=datetime.strptime("2015-09-01", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo")),
                 to_date=datetime.strptime("2015-11-30", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo")),
                 grid_boundaries_tuple=(-180, 180, -90, 90), spatial_resolution_decimals = 3,
                 country="Japan"):
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
        self.country=country

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
            cooccurrences = self.database.find_cooccurrences(user, cell_size, self.timebin_size, asGeoJSON=False)
            for occurrence in cooccurrences:
                time_bins = self.map_time_to_timebins(occurrence[1], occurrence[2])
                lng = occurrence[3]
                lat = occurrence[4]
                spatial_bin = self.calculate_spatial_bin(lng, lat)
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
        lat = int(lat * 10**self.spatial_resolution_decimals) / 10.0**self.spatial_resolution_decimals
        lng = int(lng * 10**self.spatial_resolution_decimals) / 10.0**self.spatial_resolution_decimals

        return self.database.find_cooccurrences_within_area(lng, lat, time_bin)

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
        user1_locations = self.database.get_locations_for_user(user1, self.country)
        user2_locations = self.database.get_locations_for_user(user2, self.country)
        time_bin_range = self.map_time_to_timebins(self.min_datetime, self.max_datetime)
        array_size = abs(self.GRID_MAX_LAT-self.GRID_MIN_LAT)*abs(self.GRID_MAX_LNG-self.GRID_MIN_LNG)

        for bin in tqdm(time_bin_range):
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
        cooccurrences = self.database.find_cooccurrences(user1, cell_size, self.timebin_size, useruuid2=user2, asGeoJSON=False)
        arr_leav_values = []
        for cooc in tqdm(cooccurrences):
            lng = cooc[1]
            lat = cooc[2]
            arr_leav_value = 0
            spatial_bin = self.calculate_spatial_bin(lng, lat)
            # sort timebins to reliably get previous and next timebins outside their cooc
            time_bins = sorted(cooc[3])
            # check if one of the users are in the previous timebin but not both
            previous_list = self.find_users_in_cooccurrence(lng, lat, time_bins[0]-1)
            current_list = self.find_users_in_cooccurrence(lng, lat, time_bins[0])
            next_list = self.find_users_in_cooccurrence(lng, lat, time_bins[-1]+1)

            
            number_of_new_arrivals = len(set(current_list)-set(previous_list))
            number_of_leavers = len(set(next_list)-set(current_list))

           
            if (user1 in previous_list and user2 not in previous_list) or (user1 not in previous_list and user2 in previous_list):
                # non-synchronously arrival
                arr_leav_value += 0
            else:
                # synchronous arrival
                if number_of_new_arrivals == 0:
                    arr_leav_value+=1
                else:
                    arr_leav_value += (1/(number_of_new_arrivals))

            if (user1 in next_list and user2 not in next_list) or (user1 not in next_list and user2 in next_list):
                # non-synchronously leaving
                arr_leav_value += 0
            else:
                # synchronous leaving
                if number_of_leavers == 0:
                    arr_leav_value+=1
                else:
                    arr_leav_value += (1/(number_of_leavers))
            
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
        cooccurrences = self.database.find_cooccurrences(user1, cell_size, self.timebin_size, useruuid2=user2, asGeoJSON=False)
        coocs_w_values = []
        for cooc in tqdm(cooccurrences):
            lng = cooc[1]
            lat = cooc[2]

            coocs_w_value = 0

            time_bins = cooc[3]

            for time_bin in time_bins:
                users = self.find_users_in_cooccurrence(lng, lat, time_bin)
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
    
    def find_friend_pairs(self):
        cell_size = pow(10, -self.spatial_resolution_decimals)
        user_pairs = pickle.load( open( "cooc_userPairs.p", "rb" ) )
        user_pairs.sort(key=lambda tup: tup[2])
        single_coocs = []
        for pair in tqdm(user_pairs):
            coocs = self.database.find_cooccurrences(pair[0], cell_size, self.timebin_size, useruuid2=pair[1], asGeoJSON=False)
            count = 0
            for cooc in coocs:
                timebins = cooc[3]
                user_lengths = [len(self.find_users_in_cooccurrence(cooc[1], cooc[2], bin)) for bin in timebins]
                if all(length==2 for length in user_lengths):
                    count += 1
                if count >= 5:
                    single_coocs.append(pair)
                    print(pair)
                    break
        return single_coocs

    def find_friend_pairs2(self):
        cell_size = pow(10, -self.spatial_resolution_decimals)
        user_pairs = pickle.load( open( "cooc_userPairs.p", "rb" ) )
        user_pairs.sort(key=lambda tup: tup[2])
        single_coocs = []
        for pair in user_pairs:
            coocs = self.database.find_cooccurrences(pair[0], cell_size, self.timebin_size, useruuid2=pair[1], asGeoJSON=False)
            count = 0
            for cooc in coocs:
                timebins = self.map_time_to_timebins(cooc[4], cooc[5])
                user_lengths = [len(self.find_users_in_cooccurrence(cooc[1], cooc[2], bin)) for bin in timebins]
                if all(length==2 for length in user_lengths):
                    count += 1
                if count >= 5:
                    single_coocs.append(pair)
                    print(pair)
                    break
        return single_coocs

if __name__ == '__main__':
    JAPAN_TUPLE = (120, 150, 20, 45)
    decimals = 2
    p = Predictor(60, grid_boundaries_tuple=JAPAN_TUPLE, spatial_resolution_decimals=decimals)
    #p.generate_dataset("Japan", 0.001)
    #print(p.calculate_corr("492f0a67-9a2c-40b8-8f0a-730db06abf65", "4bd3f3b1-791f-44be-8c52-0fd2195c4e62"))
    #print(p.calculate_coocs_w("492f0a67-9a2c-40b8-8f0a-730db06abf65", "4bd3f3b1-791f-44be-8c52-0fd2195c4e62"))
    #
    #print(len(p.find_users_in_cooccurrence(13.2263406245194, 55.718135067203, 521)))
    #print(timeit.timeit('p.find_users_in_cooccurrence(13.2263406245194, 55.718135067203, 521)', number=1, setup="from Predictor import Predictor;JAPAN_TUPLE = (120, 150, 20, 45);p = Predictor(60, grid_boundaries_tuple=JAPAN_TUPLE, spatial_resolution_decimals=2)"))
    #print(p.calculate_arr_leav("9b3edd01-b821-40c9-9f75-10cb32aa14b6", "3084b64d-e773-4daa-aeea-cc3b069594f3"))
    #p.find_friend_pairs()
    p.find_friend_pairs2()
    #p.calculate_arr_leav('cfd65fd1-59d5-47d7-a032-1c93bed191d6', '052db813-aab4-4317-8c4d-fb772007ff12')