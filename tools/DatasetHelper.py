#!/usr/bin/env python3
from FileLoader import FileLoader
import numpy as np
import itertools
import collections
from tqdm import tqdm


class DatasetHelper():

    def __init__(self):
        self.file_loader = FileLoader()

    def calculate_unique_cooccurrences(self, cooc_arr):
        return np.unique(cooc_arr[:, 2]).shape[0]

    def calculate_arr_leav(self, cooc_arr, loc_arr):
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences in arr_leav")
            return 0
        arr_leav_values = []
        for row in cooc_arr:
            arr_leav_value = 0

            user1_present_in_previous = loc_arr[(loc_arr[:, 0] == row[0]) & (
                loc_arr[:, 2] == row[3]-1) & (loc_arr[:, 1] == row[2])].size
            user2_present_in_previous = loc_arr[(loc_arr[:, 0] == row[1]) & (
                loc_arr[:, 2] == row[3]-1) & (loc_arr[:, 1] == row[2])].size
            if not user1_present_in_previous and not user2_present_in_previous:
                # synchronous arrival
                # finds users in previous timebin with spatial bin
                before_arrive_list = loc_arr[
                    (loc_arr[:, 2] == row[3]-1) & (loc_arr[:, 1] == row[2])][:, 0]
                # finds users in current timebin with spatial bin
                arrive_list = loc_arr[
                    (loc_arr[:, 2] == row[3]) & (loc_arr[:, 1] == row[2])][:, 0]
                num_arrivals = np.setdiff1d(
                    arrive_list, before_arrive_list, assume_unique=True).shape[0]
                if num_arrivals == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/num_arrivals)

            user1_present_in_next = loc_arr[(loc_arr[:, 0] == row[0]) & (
                loc_arr[:, 2] == row[3]+1) & (loc_arr[:, 1] == row[2])].size
            user2_present_in_next = loc_arr[(loc_arr[:, 0] == row[1]) & (
                loc_arr[:, 2] == row[3]+1) & (loc_arr[:, 1] == row[2])].size
            if not user1_present_in_next and not user1_present_in_previous:
                # synchronous leaving
                leave_list = loc_arr[
                    (loc_arr[:, 2] == row[3]) & (loc_arr[:, 1] == row[2])][:, 0]
                # finds users in current timebin with spatial bin
                after_leave_list = loc_arr[
                    (loc_arr[:, 2] == row[3]+1) & (loc_arr[:, 1] == row[2])][:, 0]
                num_leavers = np.setdiff1d(
                    after_leave_list, leave_list, assume_unique=True).shape[0]
                if num_leavers == 0:
                    arr_leav_value += 1
                else:
                    arr_leav_value += (1/num_leavers)
            arr_leav_values.append(arr_leav_value)

        return sum(arr_leav_values)/cooc_arr.shape[0]

    def calculate_coocs_w(self, cooc_arr, loc_arr):
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences for cooc_w")
            return 0
        coocs_w_values = []
        for row in cooc_arr:
            coocs_w_value = loc_arr[(loc_arr[:, 1] == row[2]) &
                                    (loc_arr[:, 2] == row[3])].shape[0]
            coocs_w_values.append(1/(coocs_w_value-1))
            print(coocs_w_value)
        return sum(coocs_w_values)/cooc_arr.shape[0]

    def calculate_diversity(self, cooc_arr):
        frequency = cooc_arr.shape[0]
        _, counts = np.unique(cooc_arr[:, 2], return_counts=True)

        shannon_entropy = -np.sum((counts/frequency) *
                                  (np.log2(counts/frequency)))
        return np.exp(shannon_entropy)

    def calculate_weighted_frequency(self, cooc_arr, loc_arr):
        weighted_frequency = 0
        spatial_bins, counts = np.unique(cooc_arr[:, 2], return_counts=True)
        for spatial_bin, count in zip(spatial_bins, counts):
            # get all locations with spatial bin
            locs_with_spatial = loc_arr[loc_arr[:, 1] == spatial_bin]
            location_entropy = 0
            for user in np.unique(locs_with_spatial[:, 0]):
                # get all locations for user
                v_lu = locs_with_spatial[locs_with_spatial[:, 0] == user]
                # get all locations for spatial bin
                v_l = locs_with_spatial
                prob = v_lu.shape[0]/v_l.shape[0]
                if prob != 0:
                    location_entropy += prob*np.log2(prob)
            location_entropy = -location_entropy
            weighted_frequency += count * np.exp(-location_entropy)
        return weighted_frequency

    def generate_cooccurrences_array(self, loc_arr):
        labels = ["useruuid1", "useruuid2", "spatial_bin", "time_bin"]
        coocs_dict = collections.defaultdict(list)
        cooccurrences = []
        for x in loc_arr:
            coocs_dict[(x[1],x[2])].append(x[0])
        for x, coocs_list in tqdm(coocs_dict.items()):
            for pair in itertools.combinations(set(coocs_list), 2):
                cooccurrences.append([pair[0],pair[1],x[0],x[1]])
        return np.array(cooccurrences)

    def generate_cooccurrences_array_old(self, loc_arr):
        unique_users = np.unique(loc_arr[:, [0]])
        labels = ["useruuid1", "useruuid2", "spatial_bin", "time_bin"]
        cooccurrences_list = []
        # generate all combinations of users
        for user_pair in tqdm(list(itertools.combinations(unique_users, 2))):
            user1 = user_pair[0]
            user2 = user_pair[1]
            # find locations for user1 and user2
            user1_arr = loc_arr[(loc_arr[:, 0] == user1)]
            user2_arr = loc_arr[(loc_arr[:, 0] == user2)]

            # extract time and spatial bin columns
            user1_arr = user1_arr[:, [1, 2]]
            user2_arr = user2_arr[:, [1, 2]]
            # create sets of of spatial, time tuples and find the intersection
            user1_set = set([tuple(x) for x in user1_arr])
            user2_set = set([tuple(x) for x in user2_arr])
            cooccurrences = [x for x in user1_set & user2_set]
            for cooc in cooccurrences:
                spatial_bin = cooc[0]
                time_bin = cooc[1]
                cooccurrences_list.append([user1, user2, spatial_bin, time_bin])
        return np.array(cooccurrences_list)
