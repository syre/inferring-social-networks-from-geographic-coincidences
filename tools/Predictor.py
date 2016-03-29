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
import sklearn.ensemble
from sklearn import cross_validation
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
    
    def generate_dataset(self):
        friend_pairs = [('a1af1eea-3d20-442d-89a3-3a17231a5046', 'b5d566b6-933e-4540-9fbf-5b17b3518993'),
                        ('971d4c06-37ce-46c2-99e1-9db4f3132189', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('8ec57410-c13d-400b-835f-c60cf34c9df8', 'a1c90af7-95d9-426f-8239-530957180051'),
                        ('9e5bb488-6bf4-4039-a10b-8b9c51357c8f', 'd859435d-bf70-47b3-a783-be5d2eae90a1'),
                        ('0a4611fa-18f3-4cfa-9557-68247f20b9af', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('16fb81b6-edd2-44de-9a83-b7b2e9dabdb1', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('5003d458-7613-4994-9bc7-ba80c29347f4', 'd3da4b5e-87f6-4320-8163-d15b9e4cb343'),
                        ('16fb81b6-edd2-44de-9a83-b7b2e9dabdb1', '81584191-955a-442f-9a0b-5931645ce00c'),
                        ('cfd65fd1-59d5-47d7-a032-1c93bed191d6', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('c8beda11-6c53-4ea3-8b30-16e32293df05', 'cfd65fd1-59d5-47d7-a032-1c93bed191d6'),
                        ('b174b5ec-b6e5-4f46-a02f-b5aa9cfd404e', 'cfd65fd1-59d5-47d7-a032-1c93bed191d6'),
                        ('441ba908-a322-41e8-88b4-39c1b0306288', '50db07e4-ba68-47d1-8a84-4e6b8abed9a3'),
                        ('9e5bb488-6bf4-4039-a10b-8b9c51357c8f', 'd859435d-bf70-47b3-a783-be5d2eae90a1'),
                        ('5b156d3d-a486-4490-9262-8726dd79e3eb', '734da756-7aed-4c5a-9315-c1cf4f8e0be7'),
                        ('5003d458-7613-4994-9bc7-ba80c29347f4', 'd3da4b5e-87f6-4320-8163-d15b9e4cb343'),
                        ('b5d566b6-933e-4540-9fbf-5b17b3518993', 'cfd65fd1-59d5-47d7-a032-1c93bed191d6'),
                        ('351c931e-3583-4247-a301-feae34a05f67', '7c357428-7ed5-4730-85fe-e82cbf00bd2a'),
                        ('982544cd-6cfb-4bd4-b86e-dfad271e2837', 'eb71290c-9a15-4d4a-b6e1-6e4e9ab4d332'),
                        ('4e2fc9bf-87c9-4b00-8bd3-e4d64db325da', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('098f0dc0-8799-4a4b-b61e-c74852c3d2b6', 'e926951f-2834-4c95-af25-2aacbca9a085'),
                        ('39a5f6f6-5c47-4fb4-9951-9e36f8b2f517', 'b57525d5-39ff-4323-bfb2-7aa8daad6960'),
                        ('b5d566b6-933e-4540-9fbf-5b17b3518993', 'd623bbd3-4091-4800-90f6-c6dc18ed97d3'),
                        ('1e0e560f-ffea-488f-ba61-895cdb3f7d63', '4daf7776-93a6-4562-b616-be458df70423'),
                        ('39a5f6f6-5c47-4fb4-9951-9e36f8b2f517', 'e6d9641e-75d2-4846-b379-d42396aff529'),
                        ('5d611833-7801-4c13-83a4-fc8f56f8c854', '82e1405e-7b7a-4786-a80e-72a826adbd2a'),
                        ('10112159-99be-41a2-b7dd-3e82ae3e0708', '945d4ad7-32e0-4a21-8e80-506b72f58a1b'),
                        ('a9574f54-a537-4d98-a901-ff9ef6d1b8dd', 'e804487b-89d1-4bca-84e6-1606ddf556af'),
                        ('441ba908-a322-41e8-88b4-39c1b0306288', '82e1405e-7b7a-4786-a80e-72a826adbd2a'),
                        ('026dd6a0-83b4-4dbb-a1cf-1e57a753ae9c', 'f7def713-1503-49f2-af7b-dd9dca7fc557'),
                        ('1f38b628-0fa0-4a6a-bc02-9fa63697afb0', '50db07e4-ba68-47d1-8a84-4e6b8abed9a3'),
                        ('8adbdca6-cfe9-40df-af2a-fca8bd3f255d', 'b252468a-6b8d-48ad-9db0-53e0a7559526'),
                        ('1e068685-c708-44a9-86ba-9e174f137163', '82e1405e-7b7a-4786-a80e-72a826adbd2a'),

                        ]

        non_friend_pairs = [('1e5139c4-fd02-44fa-afef-42efd07ed7f0', '85f8bd3b-ac83-4679-8de1-73cc3d507300'),
                            ('1e0e560f-ffea-488f-ba61-895cdb3f7d63', '8b0bdc7e-b176-417a-8d57-2f65e64f372e'),
                            ('1e068685-c708-44a9-86ba-9e174f137163', 'b3b24ce2-ec50-4f22-aa6c-9da70f6937c2'),
                            ('1e068685-c708-44a9-86ba-9e174f137163', 'b252468a-6b8d-48ad-9db0-53e0a7559526'),
                            ('16fb81b6-edd2-44de-9a83-b7b2e9dabdb1', 'cea2be4f-a53a-4a9c-9bc1-26344ad81591'),
                            ('108a7785-e4b1-4a6c-9efd-a6203d3c7f95', 'cfd65fd1-59d5-47d7-a032-1c93bed191d6'),
                            ('0a4611fa-18f3-4cfa-9557-68247f20b9af', '34b1d189-d08d-4173-a1f4-1eb51297e6d0'),
                            ('09a2aeff-8780-49d9-9a6e-1c67a51ac6f6', 'cd5ad858-bbf4-4dcb-a826-b27dc5d6f362'),
                            ('098f0dc0-8799-4a4b-b61e-c74852c3d2b6', 'ed7b8ae1-7957-46a8-8274-5268a97aaa56'),
                            ('a1c90af7-95d9-426f-8239-530957180051', 'd623bbd3-4091-4800-90f6-c6dc18ed97d3'),
                            ('a890b735-9000-4505-b148-c882e2d8d0c5', 'b789cbd0-4a86-403c-b0c3-1f6eff166e99'),
                            ('3d3b2f25-a81e-473a-9406-a8d8124a97c7', '9cd05cfb-c889-41af-a243-724c01340daf'),
                            ('4261def0-511d-4c73-a2a4-ba0140b05218', '4e2fc9bf-87c9-4b00-8bd3-e4d64db325da'),
                            ('4cae4917-0cdf-4b0f-87eb-c9efc03a4231', '8ec57410-c13d-400b-835f-c60cf34c9df8'),
                            ('81aba82f-fd28-4e7e-8a1e-3f8ce552d689', '8f454d35-049f-4bf0-9d20-1e681b558540'),
                            ('0a4611fa-18f3-4cfa-9557-68247f20b9af', '5003d458-7613-4994-9bc7-ba80c29347f4'),
                            ('4daf7776-93a6-4562-b616-be458df70423', 'e21901af-70ba-402c-9e98-92fd6e0656f6'),
                            ('6354e2e5-7d2b-4bcf-b3d5-e6cdee73ffbb', '63e89ff4-6c93-4aee-8d17-95fc8c243de2'),
                            ('81aba82f-fd28-4e7e-8a1e-3f8ce552d689', '9a8008dc-6d28-4e92-9ba7-706fc03ab944'),
                            ('cd5ad858-bbf4-4dcb-a826-b27dc5d6f362', 'da6d0040-db6c-4add-a438-4f639b0ad46e'),
                            ('d9fa2b71-027f-47cb-aac2-eb13b123d154', 'dd720486-5b6b-4140-8c5c-b74c8167c4c3'),
                            ('03c65422-9939-426b-8875-8b2ee17700ba', '9435eb49-ad44-4111-b7eb-47d43c5fb97d'),
                            ('098f0dc0-8799-4a4b-b61e-c74852c3d2b6', '512bc8d3-0210-433a-9b21-a6968b998d06'),
                            ('10112159-99be-41a2-b7dd-3e82ae3e0708', '2d0d804b-c603-4ca4-95d3-8e7c98be80a5'),
                            ('23b77597-0ab2-4085-8766-45e7d41cc1d8', 'dc2170be-0bd0-4eef-91c9-026642ff6a83'),
                            ('098f0dc0-8799-4a4b-b61e-c74852c3d2b6', 'f7def713-1503-49f2-af7b-dd9dca7fc557'),
                            ('0bd8ccc7-d782-45e8-bf02-81d624c8fa19', '3c3a2552-3602-4fed-9f0a-222f14c36934'),
                            ('0bd8ccc7-d782-45e8-bf02-81d624c8fa19', '8177d17f-c96d-4c95-babf-ddb3c612641d'),
                            ('39a5f6f6-5c47-4fb4-9951-9e36f8b2f517', '512bc8d3-0210-433a-9b21-a6968b998d06'),
                            ('4261def0-511d-4c73-a2a4-ba0140b05218', '449c005f-073e-4780-afed-831f89a0980b')
                            ]
        X = np.ndarray(shape=(len(friend_pairs)+len(non_friend_pairs),4), dtype="float")
        for index, pair in enumerate(friend_pairs):
            X[index:,0] = len(self.database.find_cooccurrences(pair[0], 0.01, 60, useruuid2=pair[1]))
            X[index:,1] = self.calculate_corr(pair[0], pair[1])
            X[index:,2] = self.calculate_arr_leav(pair[0], pair[1])
            X[index:,3] = self.calculate_coocs_w(pair[0], pair[1])

        for index, pair in enumerate(non_friend_pairs, start=len(friend_pairs)):
            X[index:,0] = len(self.database.find_cooccurrences(pair[0], 0.01, 60, useruuid2=pair[1]))
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
        if len(cooccurrences) == 0:
            return 0
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
                
                if num_users < 2:
                    print("only " + str(num_users) + " together in timebin")
                    continue
                coocs_w_value += num_users
            
            coocs_w_value /= len(time_bins)

            # 2 users is ideal thus returning highest value 1, else return lesser value proportional to amount of users
            if coocs_w_value > 2:
                coocs_w_values.append(1/(coocs_w_value-1))
        if len(cooccurrences) == 0:
            return 0
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
        cell_size = pow(10, -self.spatial_resolution_decimals)
        user_pairs = pickle.load( open( "cooc_userPairs.p", "rb" ) )
        user_pairs.sort(key=lambda tup: tup[2])
        friend_pairs = []
        nonfriend_pairs = []
        for pair in tqdm(user_pairs):
            coocs = self.database.find_cooccurrences(pair[0], cell_size, self.timebin_size, useruuid2=pair[1], asGeoJSON=False)
            count = 0
            was_pair = False
            for cooc in coocs:
                timebins = cooc[3]
                user_lengths = [len(self.find_users_in_cooccurrence(cooc[1], cooc[2], bin)) for bin in timebins]
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

    def find_friend_and_nonfriend_pairs2(self):
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
    #p.find_friend_and_nonfriend_pairs()
    p.predict()
    #p.find_friend_pairs2()
    #p.calculate_arr_leav('cfd65fd1-59d5-47d7-a032-1c93bed191d6', '052db813-aab4-4317-8c4d-fb772007ff12')