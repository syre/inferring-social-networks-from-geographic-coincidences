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
        if cooc_arr.shape[0] == 0:
            print("no cooccurrences for cooc_w")
            return 0
        coocs_w_values = []
        for row in cooc_arr:
            coocs_w_value = loc_arr[(loc_arr[:, 1] == row[2]) &
                                    (loc_arr[:, 2] == row[3])].shape[0]
            coocs_w_values.append(1/(coocs_w_value-1))
        return sum(coocs_w_values)/cooc_arr.shape[0]

    def calculate_diversity(self, cooc_arr):
        """
        Diversity quantifies how many locations the cooccurrences between two people represent.
        Can either use Shannon or Renyi Entropy, right now uses Shannon
        From inferring realworld relationships from spatiotemporal data paper p. 23.
        """
        frequency = cooc_arr.shape[0]
        _, counts = np.unique(cooc_arr[:, 2], return_counts=True)

        shannon_entropy = -np.sum((counts/frequency) *
                                  (np.log2(counts/frequency)))
        return np.exp(shannon_entropy)

    def calculate_weighted_frequency(self, cooc_arr, loc_arr):
        """
        Inferring realworld relationships from spatiotemporal data paper p. 19 and 23-25
        Tells how important co-occurrences are at non-crowded places
        """
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
        """
        Returns an array of cooccurrences with columns:
        ["useruuid1", "useruuid2", "spatial_bin", "time_bin"]
        It removes location spatial/time duplicates
        and the smallest userid is always in column useruuid1
        """
        coocs_dict = collections.defaultdict(list)
        cooccurrences = []
        for x in loc_arr:
            coocs_dict[(x[1], x[2])].append(x[0])
        for x, coocs_list in tqdm(coocs_dict.items()):
            for pair in itertools.combinations(set(coocs_list), 2):
                # make column1 always have the lowest userid
                pair = sorted(pair)
                cooccurrences.append([pair[0], pair[1], x[0], x[1]])
        return np.array(cooccurrences)