#!/usr/bin/env python3
import DatabaseHelper
import math
import datetime
from datetime import datetime, timedelta
from collections import defaultdict
from pytz import timezone
import numpy as np
from scipy import stats
import sklearn
import sklearn.ensemble
from sklearn import cross_validation
import timeit
from tqdm import tqdm
import pickle
import json
import random

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
    
    def generate_dataset(self, friend_pairs, non_friend_pairs):
        friend_pairs = random.sample(friend_pairs, 100)
        non_friend_pairs = random.sample(non_friend_pairs, 100)

        X = np.ndarray(shape=(len(friend_pairs)+len(non_friend_pairs),4), dtype="float")
        for index, pair in tqdm(enumerate(friend_pairs)):
            X[index:,0] = len(self.database.find_cooccurrences(pair[0], useruuid2=pair[1]))
            X[index:,1] = self.calculate_corr(pair[0], pair[1])
            X[index:,2] = self.calculate_arr_leav(pair[0], pair[1])
            X[index:,3] = self.calculate_coocs_w(pair[0], pair[1])

        for index, pair in tqdm(enumerate(non_friend_pairs, start=len(friend_pairs))):
            X[index:,0] = len(self.database.find_cooccurrences(pair[0], useruuid2=pair[1]))
            X[index:,1] = self.calculate_corr(pair[0], pair[1])
            X[index:,2] = self.calculate_arr_leav(pair[0], pair[1])
            X[index:,3] = self.calculate_coocs_w(pair[0], pair[1])
        
        y = np.array([1 for x in range(len(friend_pairs))] + [0 for x in range(len(non_friend_pairs))])

        return X,y

    def predict(self):
        X, y = self.generate_dataset()
        tree = sklearn.ensemble.RandomForestClassifier()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
        tree.fit(X_train, y_train)
        print(tree.score(X_test, y_test))


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
    
    def find_users_in_cooccurrence(self, spatial_bin, time_bin):
        """
        Find all users who's been in a given cooccurrence
            Arguments:
                lat {float} -- latitude
                lng {float} -- longitude
                time_bin {integer} -- time bin index
            Returns:
                list -- list of user_uuids
        
        """
        return self.database.find_cooccurrences_within_area(spatial_bin, time_bin)

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

        for time_bin in time_bin_range:
            user1_vector = np.zeros(array_size, dtype=bool)
            for location in user1_locations:
                start_time = location[1]
                end_time = location[2]
                lng = location[3]
                lat = location[4]
                if time_bin in self.map_time_to_timebins(start_time, end_time):
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

        cooccurrences = self.database.find_cooccurrences(user1, useruuid2=user2, asGeoJSON=False)
        arr_leav_values = []
        if len(cooccurrences) == 0:
            print("no cooccurrences for users: {} and {}".format(user1, user2))
            return 0

        for cooc in cooccurrences:
            spatial_bin = cooc[1]
            arr_leav_value = 0

            # sort timebins to reliably get previous and next timebins outside their cooc
            user1_time_bins = cooc[2]
            user2_time_bins = cooc[3]

            if not (min(user1_time_bins) == min(user2_time_bins)):
                # non-synchronously arrival
                arr_leav_value += 0
            else:
                # synchronous arrival
                before_arrive_list = self.find_users_in_cooccurrence(spatial_bin, min(user1_time_bins)-1)
                arrive_list = self.find_users_in_cooccurrence(spatial_bin, min(user1_time_bins))

                number_of_new_arrivals = len(set(arrive_list)-set(before_arrive_list))

                if number_of_new_arrivals == 0:
                    arr_leav_value+=1
                else:
                    arr_leav_value += (1/(number_of_new_arrivals))

            if not (max(user1_time_bins) == max(user2_time_bins)):
                # non-synchronously leaving
                arr_leav_value += 0
            else:
                # synchronous leaving
                leave_list = self.find_users_in_cooccurrence(spatial_bin, max(user1_time_bins))
                after_leave_list = self.find_users_in_cooccurrence(spatial_bin, max(user1_time_bins)+1)

                number_of_leavers = len(set(after_leave_list)-set(leave_list))
                
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
        cooccurrences = self.database.find_cooccurrences(user1, useruuid2=user2, asGeoJSON=False)
        coocs_w_values = []
        if len(cooccurrences) == 0:
            return 0
        for cooc in cooccurrences:
            lng = cooc[1]
            lat = cooc[2]

            coocs_w_value = 0
            spatial_bin = cooc[1]
            user1_time_bins = cooc[2]
            user2_time_bins = cooc[3]
            common_time_bins = set(user1_time_bins) & set(user2_time_bins)
            for time_bin in common_time_bins:
                users = self.find_users_in_cooccurrence(spatial_bin, time_bin)
                num_users = len(users)
                
                if num_users < 2:
                    print("only {} together in timebin for users: {} and {}".format(str(num_users), user1, user2))
                    continue
                coocs_w_value += num_users
            
            coocs_w_value /= len(common_time_bins)

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
    
    def find_friend_and_nonfriend_pairs(self):
        user_pairs = pickle.load( open( "cooc_userPairs.p", "rb" ) )
        user_pairs.sort(key=lambda tup: tup[2])
        friend_pairs = []
        nonfriend_pairs = []
        for pair in tqdm(user_pairs):
            coocs = self.database.find_cooccurrences(pair[0], useruuid2=pair[1], asGeoJSON=False)
            count = 0
            was_pair = False
            for cooc in coocs:
                user2_timebins = cooc[2]
                user1_timebins = cooc[3]
                timebins = set(user1_timebins) & set(user2_timebins)
                user_lengths = [len(self.find_users_in_cooccurrence(cooc[1], bin)) for bin in timebins]
                if all(length==2 for length in user_lengths):
                    count += 1
                if count >= 5:
                    friend_pairs.append(pair)
                    was_pair = True
                    print("friend-pair: " + str(pair))
                    break
            if not was_pair:
                nonfriend_pairs.append(pair)
                print("non-friend pair: " + str(pair))
        return friend_pairs, nonfriend_pairs
    
    def save_friend_and_nonfriend_pairs(self, friend_pairs, nonfriend_pairs):
        json_dict = {"friends":[], "nonfriends":[]}
        for friend in friend_pairs:
            json_dict["friends"].append({"user1":friend[0], "user2":friend[1], "count":friend[2]})
        for nonfriend in nonfriend_pairs:
            json_dict["nonfriends"].append({"user1":nonfriend[0], "user2":nonfriend[1], "count":nonfriend[2]})
        
        with open("friends_and_nonfriends.json","w+") as fp:
            json.dump(json_dict, fp)

    def load_friend_and_nonfriend_pairs(self):
        with open("friends_and_nonfriends.json", "r") as fp:
            json_dict = json.load(fp)
        friend_pairs = [(x["user1"],x["user2"],x["count"]) for x in json_dict["friends"]]
        nonfriend_pairs = [(x["user1"],x["user2"],x["count"]) for x in json_dict["nonfriends"]]
        return friend_pairs, nonfriend_pairs

    def find_friend_pairs(self):
        user_dict = defaultdict(dict)
        friends = []
        nonfriends = []
        users = self.database.get_users_in_country("Japan")
        for user in tqdm(users):
            user_dict[user] = []

        for user in tqdm(users):
            for user2 in tqdm(users, nested=True):
                if user != user2 and user not in user_dict[user2] and user2 not in user_dict[user]:
                    result = self.database.find_cooccurrences(user, [[(139.743862,35.630338), 1000]], user2, asGeoJSON=False)
                    if len(result) >=5:
                        friend_flag = False
                        count = 0
                        for cooc in result:
                            spatial_bin = cooc[1]
                            common_time_bins = list(set(cooc[2]) & set(cooc[2]))
                            user_lengths = [len(self.find_users_in_cooccurrence(spatial_bin, tbin)) for tbin in common_time_bins]

                            if all([length==2 for length in user_lengths]):
                                count+=1

                            if count>=5:
                                friend_flag = True
                                friends.append((user, user2, len(result)))
                                break

                    user_dict[user2].append(user)
                    user_dict[user].append(user2)
                    if not friend_flag:
                        nonfriends.append((user, user2))


        pickle.dump( friends, open( "friendPairs.p", "wb" ) )
        pickle.dump( nonfriends, open( "nonfriends.p", "wb" ) )

if __name__ == '__main__':
    JAPAN_TUPLE = (120, 150, 20, 45)
    decimals = 2
    p = Predictor(60)
    p.find_friend_pairs()
  #print(len(p.find_users_in_cooccurrence(13.2263406245194, 55.718135067203, 521)))
    #print(timeit.timeit('p.find_users_in_cooccurrence(13.2263406245194, 55.718135067203, 521)', number=1, setup="from Predictor import Predictor;JAPAN_TUPLE = (120, 150, 20, 45);p = Predictor(60, grid_boundaries_tuple=JAPAN_TUPLE, spatial_resolution_decimals=2)"))
    #print(p.calculate_arr_leav("9b3edd01-b821-40c9-9f75-10cb32aa14b6", "3084b64d-e773-4daa-aeea-cc3b069594f3"))
    #friends, nonfriends = p.load_friend_and_nonfriend_pairs()
    #print(friends)
    #print(nonfriends)
    #p.find_friend_and_nonfriend_pairs()
    #p.save_friend_and_nonfriend_pairs(friends, nonfriends)
    #print(len(friends))
    #print(len(nonfriends))
    #p.generate_dataset(friends, nonfriends)
    #p.predict()
    #p.find_friend_pairs2()
    #p.calculate_arr_leav('cfd65fd1-59d5-47d7-a032-1c93bed191d6', '052db813-aab4-4317-8c4d-fb772007ff12')