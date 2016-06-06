#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader
import numpy as np


class Run():
    """docstring for ClassName"""
    def __init__(self, country):
        self.filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982), 1000],
                                              [(11.9387722, 57.7058472), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.country = country
        self.file_loader = FileLoader()
        self.database_helper = DatabaseHelper()
        self.predictor = Predictor(country=country)
        self.dataset_helper = DatasetHelper()

    def update_all_data(self):

        first_period_datetime_min = "2015-09-01 00:00:00+00:00"
        first_period_time_bin_min = self.database_helper.calculate_time_bins(first_period_datetime_min)[0]
        first_period_datetime_max = "2015-09-30 23:59:59+00:00"
        first_period_time_bin_max = self.database_helper.calculate_time_bins(first_period_datetime_max)[0]
        second_period_datetime_min = "2015-10-01 00:00:00+00:00"
        second_period_time_bin_min = self.database_helper.calculate_time_bins(second_period_datetime_min)[0]
        second_period_datetime_max = "2015-10-31 23:59:59+00:00"
        second_period_time_bin_max = self.database_helper.calculate_time_bins(second_period_datetime_max)[0]
        third_period_datetime_min = "2015-11-01 00:00:00+00:00"
        third_period_time_bin_min = self.database_helper.calculate_time_bins(third_period_datetime_min)[0]
        third_period_datetime_max = "2015-11-30 23:59:59+00:00"
        third_period_time_bin_max = self.database_helper.calculate_time_bins(third_period_datetime_max)[0]

        #selected_users = ['2f8fe4e1-a3f0-4deb-ba9b-8f0470141082', '6255db24-5443-40d3-b65a-ae1b11288c8a', '7642c962-7514-4bb0-9ddd-c74ebc1b171a', '6ce6542e-e348-454c-9bcc-f9e172e860ee', '5bc06e9f-86a4-45ad-92c1-5acc409c4767', 'ccd610d3-b349-44c2-9837-6b8df74d6fd1', '9fce25df-9806-45f9-bda7-81d82124ead8', '0c4d0349-eb16-46df-9f33-4864a6717037', 'e1fa02ad-4d6d-4def-86a2-1ade8c59ee8e', '9d7f9386-f8cc-4977-b2ae-7c3382049f38', '66d1ac8f-7d29-4d2f-a241-e6bcb2c38489', '3084b64d-e773-4daa-aeea-cc3b069594f3', 'aa360897-b5ab-431e-afc2-8bcbc7f484a6', '03fe5016-6136-42ff-b10e-38be3f7961fb', '4748c3e4-8b03-477b-8324-f443694db8eb', '37abb413-d176-423c-8db6-61f253324c28', '4b19bbd7-0df5-4ae8-929d-ef3eae78fdb8', 'f3755a15-1606-4456-aa04-baab448cc9a6', 'c3a60bbd-394e-4f42-a725-f2c832c4a6df', 'b7c20d5c-94f1-4518-b9e8-7eec71f6573b', 'b8cfa133-b671-47ff-a7ba-2588daad6f25', 'e7d206e2-9caa-4c55-adec-cb3d93a1f377', 'e3a58b79-b39f-46c0-8ba6-9836030ee133', '6840e5c5-4207-46ff-9d82-97f0d88aec16', '77e7fe62-3335-421e-af4b-851d9be51961', '2d0bb1c4-c12b-45b0-8050-ada4ce1489e6', 'b3124440-b2f8-4add-9e67-ad6adf4ec501', '7f8116c9-f76c-4bbb-a9bb-9c711e2572bf', '8480e4ca-f1b8-449a-b571-5b3f3cd93e4d', 'ab65c6e5-5ee9-45d0-a352-a52c7ef9d9d6', '2a3b7273-bca2-40ec-a446-0bedfb8b1774', '392bc7bf-5336-48e3-a063-7fb4573e24c4', 'bce7fc05-fc98-494a-a2e8-12b3c1d0b592', '38da5e71-0062-45a7-8021-f90680260b61', '1fd0503d-bbd8-407f-911a-657877e119a3', '30ed8a5e-0129-4b86-9ef3-8f5ac1570434', '8d325d9f-9341-4d00-a890-2adaf412e5ca', 'eb34bbe5-fa09-42f6-a411-cdb7d3a29a20', 'c6309812-9802-4b93-8e5d-14b5eb738438', '990a13ac-d56f-4cdc-9331-57ff1cb741ff', 'f71f425d-cffb-44e5-b725-625fae878b6d', '1e4d2af4-2836-4567-a2ba-c7bf8725d8a1', '582ff5d2-0b8d-4509-94e0-950bce056c4a', '469da227-34fb-4b72-936a-b5530d26d201', '20f92378-817e-4bd2-8f19-10b8e5c58c39', '71980824-400f-4403-ace5-7d16be77d680', 'e2215ac7-2be2-45f6-ae6e-149f89a8e7f2', '4992ac55-6d61-4e70-92f6-46549205f3bf', '9df83008-ad97-4f83-a373-98fbb5b45ef2', '2bca7540-0a08-46d6-87ef-c3927066a98e', 'ae4a02c7-6919-4aa7-8df1-cde1088124ac', 'c522b3e3-ebda-49de-8684-cedc3e9b49cd', '1901e889-1c0e-42d7-a603-b1afd3232ce9']
        print("processing users, countries and locations as numpy matrix (train and test)")
        users, countries, locations = self.database_helper.generate_numpy_matrix_from_database(self.filter_places_dict[self.country])

        print("processing cooccurrences numpy array (train and test)")
        coocs = self.dataset_helper.generate_cooccurrences_array(locations)

        print("processing coocs for met in next (train)")
        coocs_met_in_next_train = np.copy(coocs)
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] <= second_period_time_bin_max]
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] > second_period_time_bin_min]

        print("processing coocs for met in next (test)")
        coocs_met_in_next_test = np.copy(coocs)
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] <= third_period_time_bin_max]
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] > third_period_time_bin_min]

        locations_train = locations[locations[:, 2] <= first_period_time_bin_max]
        locations_train = locations[locations[:, 2] > first_period_time_bin_min]
        coocs_train = coocs[coocs[:, 3] <= first_period_time_bin_max]
        coocs_train = coocs[coocs[:, 3] > first_period_time_bin_min]

        locations_test = locations[locations[:, 2] <= second_period_time_bin_max]
        locations_test = locations[locations[:, 2] > second_period_time_bin_min]
        coocs_test = coocs[coocs[:, 3] <= second_period_time_bin_max]
        coocs_test = coocs[coocs[:, 3] > second_period_time_bin_min]

        print("processing dataset for machine learning (train)")
        X_train, y_train = self.predictor.generate_dataset(users, countries, locations_train, coocs_train, coocs_met_in_next_train, first_period_datetime_min, first_period_datetime_max)#, selected_users)
        print("processing dataset for machine learning (test)")
        X_test, y_test = self.predictor.generate_dataset(users, countries, locations_test, coocs_test, coocs_met_in_next_test, second_period_datetime_min, second_period_datetime_max)#, selected_users)

        # undersampling did meet
        #train_stacked = np.hstack((X_train, y_train.reshape(-1, 1)))
        #didnt_meets = train_stacked[train_stacked[:,-1] == 0]
        #did_meets = train_stacked[train_stacked[:,-1] == 1]
        #train_stacked = np.vstack((didnt_meets[np.random.choice(didnt_meets.shape[0], 24, replace=False)],
        #                          did_meets[np.random.choice(did_meets.shape[0], 24, replace=False)]))
        #y_train = train_stacked[:, -1]
        #X_train = np.delete(train_stacked, -1, 1)
        
        # oversampling didnt meet
        #train_stacked = np.hstack((X_train, y_train.reshape(-1, 1)))
        #didnt_meets = train_stacked[train_stacked[:,-1] == 0]
        #did_meets = train_stacked[train_stacked[:,-1] == 1]
        #train_stacked = np.vstack((didnt_meets[np.random.choice(didnt_meets.shape[0], did_meets.shape[0], replace=True)],
        #                           did_meets[np.random.choice(did_meets.shape[0], did_meets.shape[0], replace=False)]))
        #y_train = train_stacked[:, -1]
        #X_train = np.delete(train_stacked, -1, 1)
        print("saving dataset")
        self.file_loader.save_x_and_y(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        self.predictor.predict(X_train, y_train, X_test, y_test)
if __name__ == '__main__':
    r = Run(country="Sweden")
    r.update_all_data()
