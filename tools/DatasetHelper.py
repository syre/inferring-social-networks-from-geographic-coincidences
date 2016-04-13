#!/usr/bin/env python3
from FileLoader import FileLoader
import numpy as np
import itertools


class DatasetHelper():

    def __init__(self):
        self.file_loader = FileLoader.FileLoader()

    def calculate_unique_cooccurrences_numpy(self, cooc_arr):
        return np.unique(cooc_arr[:, 2]).shape[0]

    def calculate_arr_leav_numpy(self, cooc_arr, loc_arr):
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

    def calculate_coocs_w_numpy(self, cooc_arr, loc_arr):
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences for cooc_w")
            return 0
        coocs_w_values = []
        for row in cooc_arr:
            coocs_w_value = loc_arr[(loc_arr[:, 1] == row[2]) &
                                    (loc_arr[:, 2] == row[3])].shape[0]
            coocs_w_values.append(1/(coocs_w_value-1))
        return sum(coocs_w_values)/cooc_arr.shape[0]

    def calculate_diversity_numpy(self, cooc_arr):
        frequency = cooc_arr.shape[0]
        _, counts = np.unique(cooc_arr[:, 2], return_counts=True)

        shannon_entropy = -np.sum((counts/frequency) *
                                  (np.log2(counts/frequency)))
        return np.exp(shannon_entropy)

    def calculate_weighted_frequency_numpy(self, cooc_arr, loc_arr):
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

    def generate_cooccurrences_array_numpy(self, arr):
        unique_users = np.unique(arr[:, [0]])
        labels = ["useruuid1", "useruuid2", "time_bin", "spatial_bin"]
        cooccurrences_arr = np.ndarray(shape=(0, 4))
        # generate all combinations of users
        for user_pair in itertools.combinations(unique_users, 2):
            user1 = user_pair[0]
            user2 = user_pair[1]
            # find locations for user1 and user2
            user1_arr = arr[(arr[:, 0] == user1)]
            user2_arr = arr[(arr[:, 0] == user2)]

            # extract time and spatial bin columns
            user1_arr = user1_arr[:, [1, 2]]
            user2_arr = user2_arr[:, [1, 2]]
            # retrieve indexes where rows are identical
            user1_indexes = np.unique(np.array(np.all(
                (user1_arr[:, None, :] == user2_arr[None, :, :]), axis=-1).nonzero()).T[:, [0]])
            cooccurrences = user1_arr[user1_indexes]
            user1_col = np.empty(shape=(cooccurrences.shape[0], 1))
            user2_col = np.empty(shape=(cooccurrences.shape[0], 1))
            user1_col.fill(user1)
            user2_col.fill(user2)
            cooccurrences = np.hstack(
                (np.column_stack((user1_col, user2_col)), cooccurrences))
            cooccurrences_arr = np.vstack((cooccurrences_arr, cooccurrences))
        return cooccurrences_arr
