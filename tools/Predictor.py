#!/usr/bin/env python3
import DatabaseHelper
import math
import datetime
from datetime import datetime, timedelta
import collections
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
import itertools

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
        #print("GRID_MIN_LNG = {}\nGRID_MAX_LNG = {}\nGRID_MIN_LAT = {}\nGRID_MAX_LAT = {}\n------------------".
        #    format(self.GRID_MIN_LNG, self.GRID_MAX_LNG, self.GRID_MIN_LAT, self.GRID_MAX_LAT))
    
    def generate_dataset(self, friend_pairs, non_friend_pairs, friend_size = None, nonfriend_size = None):
        users, countries, locations_arr = d.load_numpy_matrix()
        japan_arr = locations_arr[np.in1d([locations_arr[:,3]], [countries["Japan"]])]
        with open("cooccurrences.npy", "rb") as f:
            coocs = np.load(f)

        if friend_size:
            friend_pairs = random.sample(friend_pairs, friend_size)
        if nonfriend_size:
            non_friend_pairs = random.sample(non_friend_pairs, nonfriend_size)
        
        X = np.ndarray(shape=(len(friend_pairs)+len(non_friend_pairs),6), dtype="float")

        for index, pair in tqdm(enumerate(friend_pairs)):
            user1 = users[pair[0]]
            user2 = users[pair[1]]

            pair1_coocs = coocs[(coocs[:,0] == user1) & (coocs[:,1] == user2)]
            pair2_coocs = coocs[(coocs[:,0] == user2) & (coocs[:,1] == user1)]
            pair_coocs = np.vstack((pair1_coocs, pair2_coocs))
            #X[index:,0] = pair_coocs.shape[0]
            X[index:,1] = self.calculate_arr_leav_numpy(pair_coocs, japan_arr)
            X[index,2] = self.calculate_diversity_numpy(pair_coocs)
            X[index,3] = self.calculate_unique_cooccurrences_numpy(pair_coocs)
            X[index,4] = self.calculate_weighted_frequency_numpy(pair_coocs, japan_arr)
            X[index:,5] = self.calculate_coocs_w_numpy(pair_coocs, japan_arr)
            #X[index:,3] = self.calculate_corr(pair[0], pair[1])
            

        for index, pair in tqdm(enumerate(non_friend_pairs, start=len(friend_pairs))):
            user1 = users[pair[0]]
            user2 = users[pair[1]]
            pair1_coocs = coocs[(coocs[:,0] == user1) & (coocs[:,1] == user2)]
            pair2_coocs = coocs[(coocs[:,0] == user2) & (coocs[:,1] == user1)]
            pair_coocs = np.vstack((pair1_coocs, pair2_coocs))
            X[index:,0] = pair_coocs.shape[0]
            X[index:,1] = self.calculate_arr_leav_numpy(pair_coocs, japan_arr)
            X[index,2] = self.calculate_diversity_numpy(pair_coocs)
            X[index,3] = self.calculate_unique_cooccurrences_numpy(pair_coocs)
            X[index,4] = self.calculate_weighted_frequency_numpy(pair_coocs, japan_arr)
            X[index:,5] = self.calculate_coocs_w_numpy(pair_coocs, japan_arr)
            #X[index:,3] = self.calculate_corr(pair[0], pair[1])
        
        y = np.array([1 for x in range(len(friend_pairs))] + [0 for x in range(len(non_friend_pairs))])

        return X,y

    def predict(self, X, y):
        tree = sklearn.ensemble.RandomForestClassifier()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
        tree.fit(X_train, y_train)
        print(tree.score(X_test, y_test))
            
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

    def calculate_unique_cooccurrences(self, cooccurrences):
        """
        Calculates how many unique spatial bins they have had cooccurrences in
        """

        return len(set([cooc[1] for cooc in cooccurrences]))

    def calculate_arr_leav(self, cooccurrences):
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

        arr_leav_values = []
        if len(cooccurrences) == 0:
            print("no cooccurrences in arr_leav")
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


    def calculate_coocs_w(self, cooccurrences):
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
                    print("only {} together in timebin, coocs_w".format(str(num_users)))
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

    def calculate_diversity(self, cooccurrences):
        """
        Diversity quantifies how many locations the cooccurrences between two people represent.
        Can either use Shannon or Renyi Entropy, right now uses Shannon
        From inferring realworld relationships from spatiotemporal data paper p. 23.
        """

        frequency = len(cooccurrences)
        spatial_bins_counts = collections.Counter([cooc[1] for cooc in cooccurrences])

        shannon_entropy = -sum([(count/frequency)*math.log(count/frequency, 2) for _,count in spatial_bins_counts.items()])

        return np.exp(shannon_entropy)
    
    def save_x_and_y(self, x, y):
        with open( "datasetX.pickle", "wb" ) as fp:
            pickle.dump(x, fp)
        with open( "datasetY.pickle", "wb" ) as fp:
            pickle.dump(y, fp)
    
    def load_x_and_y(self):
        with open( "datasetX.pickle", "rb" ) as fp:
            x = pickle.load(fp)
        with open( "datasetY.pickle", "rb" ) as fp:
            y = pickle.load(fp)
        return x, y

    
    def calculate_weighted_frequency(self, cooccurrences):
        """
        Inferring realworld relationships from spatiotemporal data paper p. 19 and 23-25
        Tells how important co-occurrences are at non-crowded places
        """

        spatial_bins_counts = collections.Counter([cooc[1] for cooc in cooccurrences])

        weighted_frequency = 0
        for spatial_bin, count in spatial_bins_counts.items():
            unique_users = self.database.find_cooccurrences_within_area(spatial_bin)
            location_entropy = 0
            for user in unique_users:
                v_lu = self.database.get_locations_for_user(user, spatial_bin=spatial_bin)
                v_l = self.database.find_number_of_records_for_location(spatial_bin)
                prob = len(v_lu)/len(v_l)
                if prob != 0:
                    location_entropy += prob*math.log(prob, 2)
            location_entropy = -location_entropy
            weighted_frequency += count * np.exp(-location_entropy)
        return weighted_frequency

    def save_friend_and_nonfriend_pairs(self, friend_pairs, nonfriend_pairs):
        with open( "friend_pairs.pickle", "wb" ) as fp:
            pickle.dump(friend_pairs, fp)
        with open( "nonfriend_pairs.pickle", "wb" ) as fp:
            pickle.dump(nonfriend_pairs, fp)

    def load_friend_and_nonfriend_pairs(self):
        with open( "friend_pairs.pickle", "rb" ) as fp:
            friend_pairs = pickle.load(fp)
        with open( "nonfriend_pairs.pickle", "rb" ) as fp:
            nonfriend_pairs = pickle.load(fp)

        return friend_pairs, nonfriend_pairs

    def find_friend_and_nonfriend_pairs(self, ratio=0.05):
        friends = []
        nonfriends = []
        users = self.database.get_users_in_country("Japan")
        for pair in tqdm(itertools.combinations(users, 2)):
            result = self.database.find_cooccurrences(pair[0], [[(139.743862,35.630338), 1000]], pair[1], asGeoJSON=False)
            no_coocs = len(result)
            isFriends = False
            if no_coocs >=5:
                count = 0
                for cooc in result:
                    spatial_bin = cooc[1]
                    common_time_bins = list(set(cooc[2]) & set(cooc[2]))
                    user_lengths = [len(self.find_users_in_cooccurrence(spatial_bin, tbin)) for tbin in common_time_bins]
                    if all([length==2 for length in user_lengths]):
                        count+=1
                    if (count/no_coocs) >= ratio:
                        isFriends = True
                        friends.append((pair[0],pair[1], no_coocs))
                        break
            if not isFriends and no_coocs > 0:
                nonfriends.append((pair[0], pair[1]))
        return friends, nonfriends

    
    def calculate_unique_cooccurrences_numpy(self, cooc_arr):
        return np.unique(cooc_arr[:,2]).shape[0]

    def calculate_arr_leav_numpy(self, cooc_arr, loc_arr):
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences in arr_leav")
            return 0
        arr_leav_values = []
        for row in cooc_arr:
            arr_leav_value = 0
            
            user1_present_in_previous = loc_arr[(loc_arr[:,0] == row[0]) & (loc_arr[:,2] == row[3]-1) & (loc_arr[:,1] == row[2])].size
            user2_present_in_previous = loc_arr[(loc_arr[:,0] == row[1]) & (loc_arr[:,2] == row[3]-1) & (loc_arr[:,1] == row[2])].size
            if not user1_present_in_previous and not user2_present_in_previous:
                # synchronous arrival
                # finds users in previous timebin with spatial bin
                before_arrive_list = loc_arr[(loc_arr[:,2] == row[3]-1) & (loc_arr[:,1] == row[2])][:,0]
                # finds users in current timebin with spatial bin
                arrive_list = loc_arr[(loc_arr[:,2] == row[3]) & (loc_arr[:,1] == row[2])][:,0]
                num_arrivals = np.setdiff1d(arrive_list, before_arrive_list, assume_unique=True).shape[0]
                if num_arrivals == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/num_arrivals)

            user1_present_in_next = loc_arr[(loc_arr[:,0] == row[0]) & (loc_arr[:,2] == row[3]+1) & (loc_arr[:,1] == row[2])].size
            user2_present_in_next = loc_arr[(loc_arr[:,0] == row[1]) & (loc_arr[:,2] == row[3]+1) & (loc_arr[:,1] == row[2])].size
            if not user1_present_in_next and not user1_present_in_previous:
                # synchronous leaving
                leave_list = loc_arr[(loc_arr[:,2] == row[3]) & (loc_arr[:,1] == row[2])][:,0]
                # finds users in current timebin with spatial bin
                after_leave_list = loc_arr[(loc_arr[:,2] == row[3]+1) & (loc_arr[:,1] == row[2])][:,0]
                num_leavers = np.setdiff1d(after_leave_list, leave_list, assume_unique=True).shape[0]
                if num_leavers == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/num_leavers)
            arr_leav_values.append(arr_leav_value)

        return sum(arr_leav_values)/cooc_arr.shape[0]

    def calculate_coocs_w_numpy(self, cooc_arr, loc_arr):
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences for cooc_w")
            return 0
        coocs_w_values = []
        for row in cooc_arr:
            coocs_w_value = loc_arr[(loc_arr[:,1] == row[2]) & (loc_arr[:,2] == row[3])].shape[0]
            coocs_w_values.append(1/(coocs_w_value-1))
        return sum(coocs_w_values)/cooc_arr.shape[0]
    
    def calculate_diversity_numpy(self, cooc_arr):
        frequency = cooc_arr.shape[0]
        _, counts = np.unique(cooc_arr[:,2], return_counts=True)

        shannon_entropy = -np.sum((counts/frequency)*(np.log2(counts/frequency)))
        return np.exp(shannon_entropy)

    def calculate_weighted_frequency_numpy(self, cooc_arr, loc_arr):
        weighted_frequency = 0
        spatial_bins, counts = np.unique(cooc_arr[:,2], return_counts=True)
        for spatial_bin, count in zip(spatial_bins, counts):
            # get all locations with spatial bin
            locs_with_spatial = loc_arr[loc_arr[:,1] == spatial_bin]
            location_entropy = 0
            for user in np.unique(locs_with_spatial[:,0]):
                # get all locations for user
                v_lu = locs_with_spatial[locs_with_spatial[:,0] == user]
                # get all locations for spatial bin
                v_l = locs_with_spatial
                prob = v_lu.shape[0]/v_l.shape[0]
                if prob != 0:
                    location_entropy += prob*np.log2(prob)
            location_entropy = -location_entropy
            weighted_frequency += count * np.exp(-location_entropy)
        return weighted_frequency

if __name__ == '__main__':
    #JAPAN_TUPLE = (120, 150, 20, 45)
    #decimals = 2
    p = Predictor(60)
    d = DatabaseHelper.DatabaseHelper()
    #users, countries, locations_arr = d.load_numpy_matrix()
    locations_labels = ["user", "spatial_bin", "time_bin", "country"]
    cooccurrences_labels = ["user1", "user2", "spatial_bin", "time_bin"]
    #friends, nonfriends = p.find_friend_and_nonfriend_pairs()
    #p.save_friend_and_nonfriend_pairs(friends, nonfriends)
    friends, nonfriends = p.load_friend_and_nonfriend_pairs()
    X, y = p.generate_dataset(friends, nonfriends, 100, 100)
    print(X,y)
    p.predict(X,y)
    #japan_arr = locations_arr[np.in1d([locations_arr[:,3]], [countries["Japan"]])]
    #with open("cooccurrences.npy", "rb") as f:
    #        cooccurrences = np.load(f)
    #print(p.calculate_unique_cooccurrences_numpy(cooccurrences))
    #print(len(cooccurrences))
    #print(p.calculate_diversity_numpy(cooccurrences))
    #print(p.calculate_weighted_frequency_numpy(cooccurrences, locations_arr))
    #print(p.calculate_arr_leave_numpy(cooccurrences, locations_arr))
    #print(p.calculate_coocs_w(cooccurrences, locations_arr))