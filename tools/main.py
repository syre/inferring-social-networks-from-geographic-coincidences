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
                                              [(17.9529121, 59.4050982), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.country = country
        self.file_loader = FileLoader()
        self.database_helper = DatabaseHelper()
        self.predictor = Predictor(country=country)
        self.dataset_helper = DatasetHelper()

    def update_all_data(self):

        first_period_datetime_min = "2015-09-01 00:00:00+00:00"
        first_period_time_bin_min = self.database_helper.calculate_time_bins(first_period_datetime_min)[0]
        first_period_datetime_max = "2015-09-09 23:59:59+00:00"
        first_period_time_bin_max = self.database_helper.calculate_time_bins(first_period_datetime_max)[0]
        second_period_datetime_min = "2015-09-10 00:00:00+00:00"
        second_period_time_bin_min = self.database_helper.calculate_time_bins(second_period_datetime_min)[0]
        second_period_datetime_max = "2015-09-19 23:59:59+00:00"
        second_period_time_bin_max = self.database_helper.calculate_time_bins(second_period_datetime_max)[0]
        third_period_datetime_min = "2015-09-20 00:00:00+00:00"
        third_period_time_bin_min = self.database_helper.calculate_time_bins(third_period_datetime_min)[0]
        third_period_datetime_max = "2015-09-30 23:59:59+00:00"
        third_period_time_bin_max = self.database_helper.calculate_time_bins(third_period_datetime_max)[0]

        selected_users = ['91f38f33-bae3-4f7f-84e1-c80ff245ece0', '90522cae-ba2e-4677-a6e8-04ab2ddb64ed', '8e9e90bc-1817-48d5-a4c8-1e3c4637262a', 'b992b237-e563-48d7-b958-2b3e16620846', '3f35f78a-18b2-4937-bc7c-ffbf76907fc4', '26daf874-26f3-4dfc-8679-9c15aecbc18d', '1d52cba4-76f3-48d6-a8f6-408dbfeaaa45', '9ad38949-b311-47db-9195-bdbfde5f78fb', '6d4e2221-8099-4ef6-9e96-740963f74983', '17110d9c-093b-41de-b536-46a601df8786', '0c4d0349-eb16-46df-9f33-4864a6717037', '990a13ac-d56f-4cdc-9331-57ff1cb741ff', '9ad3eec3-ab95-4bf6-8968-75f20f8428a7', '2ea9c1e4-aac1-4b46-b63a-d172b13f27f0', '4748c3e4-8b03-477b-8324-f443694db8eb', '50a6e71b-59fc-41ed-9600-90efc47ee9e5', '9153eb0a-9e2d-461a-b5f9-f29ebde1fa8a', '2176c545-7cdf-4524-a0d5-0e421ee06f32', '89432c35-bdfb-4ac3-b319-735c25bffdfb', 'b7c20d5c-94f1-4518-b9e8-7eec71f6573b', '22d4e442-b9e4-4b11-9807-a2c246461619', '457c0486-fce4-443c-831c-52b3c2ec7a52', '390dc1e7-bce5-4704-9fb4-1c7e034efad2', 'c3a60bbd-394e-4f42-a725-f2c832c4a6df', 'ac9ff428-168b-432e-a6ae-f5571c3711bf', '77e7fe62-3335-421e-af4b-851d9be51961', '9df83008-ad97-4f83-a373-98fbb5b45ef2', '0bf09cdf-77a0-40f9-93a2-a4f5c9332f4d', '4b19bbd7-0df5-4ae8-929d-ef3eae78fdb8', '84bda8c6-c091-403d-87a3-eef45310cd6e', '84c54409-f489-4897-ae67-d4f368f40274', 'e3a58b79-b39f-46c0-8ba6-9836030ee133', '7642c962-7514-4bb0-9ddd-c74ebc1b171a', '8480e4ca-f1b8-449a-b571-5b3f3cd93e4d', 'fcc9196f-b23c-4839-b1c0-a853e3b35c8b', '1901e889-1c0e-42d7-a603-b1afd3232ce9', 'eb34bbe5-fa09-42f6-a411-cdb7d3a29a20', '6ce6542e-e348-454c-9bcc-f9e172e860ee', '6255db24-5443-40d3-b65a-ae1b11288c8a', '4f25e331-9fd1-4bce-a0a8-03f1ef7331c8', 'e7d206e2-9caa-4c55-adec-cb3d93a1f377', '1e4d2af4-2836-4567-a2ba-c7bf8725d8a1', '30ed8a5e-0129-4b86-9ef3-8f5ac1570434', '5f6d1c8f-e326-43b3-aedf-63cf9910ffd2', '2a3b7273-bca2-40ec-a446-0bedfb8b1774', '469da227-34fb-4b72-936a-b5530d26d201', 'f3755a15-1606-4456-aa04-baab448cc9a6', '9d7f9386-f8cc-4977-b2ae-7c3382049f38', 'd3e468a2-a6ee-48ef-b386-98bb100700f9', '38da5e71-0062-45a7-8021-f90680260b61', '5bc06e9f-86a4-45ad-92c1-5acc409c4767', 'ccd610d3-b349-44c2-9837-6b8df74d6fd1', 'ae4a02c7-6919-4aa7-8df1-cde1088124ac', '071c08fe-3fe2-49c6-ac1f-21f93cd1a87b', '2f8fe4e1-a3f0-4deb-ba9b-8f0470141082', '2bca7540-0a08-46d6-87ef-c3927066a98e', '3084b64d-e773-4daa-aeea-cc3b069594f3', '2d0bb1c4-c12b-45b0-8050-ada4ce1489e6', 'bce7fc05-fc98-494a-a2e8-12b3c1d0b592', '392bc7bf-5336-48e3-a063-7fb4573e24c4', 'f71f425d-cffb-44e5-b725-625fae878b6d', '20f92378-817e-4bd2-8f19-10b8e5c58c39', 'e2215ac7-2be2-45f6-ae6e-149f89a8e7f2', 'c6309812-9802-4b93-8e5d-14b5eb738438', '4992ac55-6d61-4e70-92f6-46549205f3bf', '0900bf92-8e62-478a-9657-c69a9388e46c', '71980824-400f-4403-ace5-7d16be77d680', '37abb413-d176-423c-8db6-61f253324c28', 'b3124440-b2f8-4add-9e67-ad6adf4ec501', '7f8116c9-f76c-4bbb-a9bb-9c711e2572bf', 'ab65c6e5-5ee9-45d0-a352-a52c7ef9d9d6', '582ff5d2-0b8d-4509-94e0-950bce056c4a', '543036c6-93ab-40a6-9668-20a70d021cdf', 'b8cfa133-b671-47ff-a7ba-2588daad6f25', '1fd0503d-bbd8-407f-911a-657877e119a3', '66d1ac8f-7d29-4d2f-a241-e6bcb2c38489', '6840e5c5-4207-46ff-9d82-97f0d88aec16', '105b7c5e-9d8a-4f65-b1d2-82bfa1e126e9', '9fce25df-9806-45f9-bda7-81d82124ead8', 'e1fa02ad-4d6d-4def-86a2-1ade8c59ee8e', 'c522b3e3-ebda-49de-8684-cedc3e9b49cd', 'aa360897-b5ab-431e-afc2-8bcbc7f484a6', '03fe5016-6136-42ff-b10e-38be3f7961fb', '8d325d9f-9341-4d00-a890-2adaf412e5ca']
        print("processing users, countries and locations as numpy matrix (train)")
        users_train, countries_train, locations_train = self.database_helper.generate_numpy_matrix_from_database()
        self.file_loader.save_numpy_matrix_train(users_train, countries_train, locations_train)

        print("processing users, countries and locations as numpy matrix (test)")
        users_test, countries_test, locations_test = self.database_helper.generate_numpy_matrix_from_database(self.filter_places_dict[self.country])
        self.file_loader.save_numpy_matrix_test(users_test, countries_test, locations_test)

        print("now: {}, before: {}".format(len([users_train[u] for u in selected_users if u in users_train]), len(selected_users)))
        print("now: {}, before: {}".format(len([users_test[u] for u in selected_users if u in users_test]), len(selected_users)))
        #locations_train = locations_train[np.in1d(locations_train[:, 0], [users_train[u] for u in selected_users if u in users_train])]
        #locations_test = locations_test[np.in1d(locations_test[:, 0], [users_test[u] for u in selected_users if u in users_test])]

        print("processing cooccurrences numpy array (train)")
        coocs_train = self.dataset_helper.generate_cooccurrences_array(locations_train)
        self.file_loader.save_cooccurrences_train(coocs_train)
        coocs_train = self.file_loader.load_cooccurrences_train()

        print("processing cooccurrences numpy array (test)")
        coocs_test = self.dataset_helper.generate_cooccurrences_array(locations_test)
        self.file_loader.save_cooccurrences_test(coocs_test)
        coocs_test = self.file_loader.load_cooccurrences_test()


        print("processing coocs for met in next (train)")
        coocs_met_in_next_train = np.copy(coocs_train)
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] <= second_period_time_bin_max]
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] > second_period_time_bin_min]

        print("finding met in next people (train)")
        #met_in_next_train = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_train)
        print("saving met in next people (train)")
        self.file_loader.save_met_in_next_train(coocs_met_in_next_train)

        print("processing coocs for met in next (test)")
        coocs_met_in_next_test = np.copy(coocs_test)
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] <= third_period_time_bin_max]
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] > third_period_time_bin_min]

        print("finding met in next people (test)")
        #met_in_next_test = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_test)
        print("saving met in next people (test)")
        self.file_loader.save_met_in_next_test(coocs_met_in_next_test)


        print("processing dataset for machine learning (train)")
        X_train, y_train = self.predictor.generate_dataset(users_train, countries_train, locations_train, coocs_train, coocs_met_in_next_train, first_period_datetime_min, first_period_datetime_max, selected_users)
        print("processing dataset for machine learning (test)")
        X_test, y_test = self.predictor.generate_dataset(users_test, countries_test, locations_test, coocs_test, coocs_met_in_next_test, second_period_datetime_min, second_period_datetime_max, selected_users)

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
