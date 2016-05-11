#!/usr/bin/env python3
from DatabaseHelper import DatabaseHelper
from Predictor import Predictor
from DatasetHelper import DatasetHelper
from FileLoader import FileLoader
import numpy as np

file_loader = FileLoader()
database_helper = DatabaseHelper()
predictor = Predictor()
dataset_helper = DatasetHelper()


class Run(object):
    """docstring for ClassName"""
    def __init__(self, country="Japan"):
        self.filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982), 1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.country = country
        self.file_loader = FileLoader()
        self.database_helper = DatabaseHelper()
        self.predictor = Predictor(country=country)
        self.dataset_helper = DatasetHelper()

    def update_all_data(self):
        # sept_min_datetime = "2015-09-01 00:00:00+00:00"
        # sept_min_time_bin = database_helper.calculate_time_bins(sept_min_datetime, sept_min_datetime)[0]
        # sept_max_datetime = "2015-09-30 23:59:59+00:00"
        # sept_max_time_bin = database_helper.calculate_time_bins(sept_max_datetime, sept_max_datetime)[0]
        # oct_min_datetime = "2015-10-01 00:00:00+00:00"
        # oct_min_time_bin = database_helper.calculate_time_bins(oct_min_datetime, oct_min_datetime)[0]
        # oct_max_datetime = "2015-10-31 23:59:59+00:00"
        # oct_max_time_bin = database_helper.calculate_time_bins(oct_max_datetime, oct_max_datetime)[0]
        # nov_min_datetime = "2015-11-01 00:00:00+00:00"
        # nov_min_time_bin = database_helper.calculate_time_bins(nov_min_datetime, nov_min_datetime)[0]
        # nov_max_datetime = "2015-11-30 23:59:59+00:00"
        # nov_max_time_bin = database_helper.calculate_time_bins(nov_max_datetime, nov_max_datetime)[0]

        first_period_datetime_min = "2015-10-01 00:00:00+00:00"
        first_period_time_bin_min = database_helper.calculate_time_bins(first_period_datetime_min)[0]
        first_period_datetime_max = "2015-10-09 23:59:59+00:00"
        first_period_time_bin_max = database_helper.calculate_time_bins(first_period_datetime_max)[0]
        second_period_datetime_min = "2015-10-10 00:00:00+00:00"
        second_period_time_bin_min = database_helper.calculate_time_bins(second_period_datetime_min)[0]
        second_period_datetime_max = "2015-10-19 23:59:59+00:00"
        second_period_time_bin_max = database_helper.calculate_time_bins(second_period_datetime_max)[0]
        third_period_datetime_min = "2015-10-20 00:00:00+00:00"
        third_period_time_bin_min = database_helper.calculate_time_bins(third_period_datetime_min)[0]
        third_period_datetime_max = "2015-10-30 23:59:59+00:00"
        third_period_time_bin_max = database_helper.calculate_time_bins(third_period_datetime_max)[0]

        selected_users = ['38da5e71-0062-45a7-8021-f90680260b61', '4ed45839-d8a7-40df-9c93-2554201f62ca', '458f17fa-126d-4079-a2d3-058a9ce2f57c', '69935fdf-9b43-417f-93df-a6f707e8b43f', 'd5a6b8e9-8ae5-4e9d-94d5-2c1865ad2e44', '4b19bbd7-0df5-4ae8-929d-ef3eae78fdb8', '4ccc2f16-5a33-4fca-a644-dc5c75a3deaa', 'b992b237-e563-48d7-b958-2b3e16620846', '3084b64d-e773-4daa-aeea-cc3b069594f3', '463b7dbb-ef29-49d0-a240-41baceea128f', '6255db24-5443-40d3-b65a-ae1b11288c8a', 'c6309812-9802-4b93-8e5d-14b5eb738438', '489e1e7c-c208-4601-b56a-287981417efe', '45691b60-e48d-4c08-9547-26223bdd7134', '0c5552ca-09c6-4cbf-ae0c-5969c3ad9983', '01caa88b-fa3c-4b03-b3e9-2b7a14beb278', 'eb34bbe5-fa09-42f6-a411-cdb7d3a29a20', 'ac9ff428-168b-432e-a6ae-f5571c3711bf', '01959ac9-8909-4d95-9557-f5b531a7f331', '2abf30d9-55f6-454e-854d-786f037b619c', '26daf874-26f3-4dfc-8679-9c15aecbc18d', 'e1fa02ad-4d6d-4def-86a2-1ade8c59ee8e', '0a21780a-1869-4cec-8ccf-50f57f5c7797', '8480e4ca-f1b8-449a-b571-5b3f3cd93e4d', 'aba7356f-77e9-4b57-8600-65f0fd479e10', 'ccd610d3-b349-44c2-9837-6b8df74d6fd1', '5a546365-f979-4f2c-a425-bddd93e667ee', '48292fe8-5d65-47f6-8c73-c23aa7030e03', '9b3edd01-b821-40c9-9f75-10cb32aa14b6', '64b6eb39-dd10-4a38-9bca-f0c90919b14c', '4992ac55-6d61-4e70-92f6-46549205f3bf', '8e9e90bc-1817-48d5-a4c8-1e3c4637262a', 'f4f32177-8c2c-418d-834c-0cde35f40cee', 'f23fd74a-f94f-4182-abaa-0ab8fb3e4a4f', '071c08fe-3fe2-49c6-ac1f-21f93cd1a87b', '7a749f71-6ccc-45bd-b4a2-782d3eb995ba', '66d1ac8f-7d29-4d2f-a241-e6bcb2c38489', 'fcc9196f-b23c-4839-b1c0-a853e3b35c8b', '163459e1-e9fd-4531-9382-b863a49adbf0', '6ce6542e-e348-454c-9bcc-f9e172e860ee', '1274307c-cd49-4f69-9cfd-9a598c426cfb', '6d4e2221-8099-4ef6-9e96-740963f74983', '6d20e7db-78f8-484c-bbf3-b5ae1f7e6b3b', '6c71df2e-54cb-489e-b110-8bb3234cfa0e', '0c4d0349-eb16-46df-9f33-4864a6717037', 'f509d570-c7c7-4cb3-9b86-b02836cac466', '0f8dd08e-7eb5-4614-aad5-cd4f9723f79c', '0d8bca9b-a4e6-4668-ba43-9521b6cb4f1e', '018b3a42-6ad5-4577-b2d8-341159eff9be', '105b7c5e-9d8a-4f65-b1d2-82bfa1e126e9', '9a9b1ef8-c48b-4105-b4fe-ab04acd3d0ee', 'e2215ac7-2be2-45f6-ae6e-149f89a8e7f2', '4bd3f3b1-791f-44be-8c52-0fd2195c4e62', '9df83008-ad97-4f83-a373-98fbb5b45ef2', '492f0a67-9a2c-40b8-8f0a-730db06abf65', 'a12f171b-76d9-4b58-a04f-9833bf10d2a3', 'a1eed029-bf59-48d1-919d-5689f6523426', 'a21d8502-c5cb-4eb5-8b19-b38ea74e9294', 'a221e5b4-ce58-49ba-9385-6b544cc8a5ae', 'a6c0e224-508b-4c28-a549-b85bd40ab770', 'aa360897-b5ab-431e-afc2-8bcbc7f484a6', 'ab65c6e5-5ee9-45d0-a352-a52c7ef9d9d6', '420b91b6-5c78-4fe6-af7c-9795edd10c0e', '3ee183a4-919a-4c67-bf84-6f196897a906', '3b48d329-f8c3-4fed-adcf-ef0aa81bbf2f', 'b0e1364a-9f2e-4b2b-8125-bd6176c16384', 'e3a58b79-b39f-46c0-8ba6-9836030ee133', '963d56e9-c390-49ad-ba83-d4c6574f676a', '36866473-4b75-4a52-a5a5-65ca6326aa04', '90522cae-ba2e-4677-a6e8-04ab2ddb64ed', '30fe0f1f-296f-4d4b-a4ef-93eaf8e05169', '8d325d9f-9341-4d00-a890-2adaf412e5ca', '8d03174e-311a-4b9f-9863-7d99f4fccd57', 'cc38817a-eb2c-448d-9194-595e210543f4', '2ddb668d-0c98-4258-844e-7e790ea65aba', '543036c6-93ab-40a6-9668-20a70d021cdf', '994b3bcf-a892-493e-b9f7-8477dec24cd5', 'd6e5fa22-27dd-47cb-acbe-eae33d029ae3', '29ec0b2f-dcc9-43e2-9421-ddada03513dd', 'd81f57bb-0925-4869-b844-8d99bf55337c', '5909445d-e68c-451f-baee-d108ca32c8cd', 'dffcd105-5c2e-41ed-848f-a20495571642', 'b3124440-b2f8-4add-9e67-ad6adf4ec501', '37abb413-d176-423c-8db6-61f253324c28']
        print("processing users, countries and locations as numpy matrix (train)")
        users_train, countries_train, locations_train = self.database_helper.generate_numpy_matrix_from_database()
        file_loader.save_numpy_matrix_train(users_train, countries_train, locations_train)

        print("processing users, countries and locations as numpy matrix (test)")
        users_test, countries_test, locations_test = self.database_helper.generate_numpy_matrix_from_database(self.filter_places_dict[self.country])
        file_loader.save_numpy_matrix_test(users_test, countries_test, locations_test)

        print("now: {}, before: {}".format(len([users_train[u] for u in selected_users if u in users_train]), len(selected_users)))
        print("now: {}, before: {}".format(len([users_test[u] for u in selected_users if u in users_test]), len(selected_users)))
        #locations_train = locations_train[np.in1d(locations_train[:, 0], [users_train[u] for u in selected_users if u in users_train])]
        #locations_test = locations_test[np.in1d(locations_test[:, 0], [users_test[u] for u in selected_users if u in users_test])]

        print("processing cooccurrences numpy array (train)")
        coocs_train = dataset_helper.generate_cooccurrences_array(locations_train)
        file_loader.save_cooccurrences_train(coocs_train)
        coocs_train = file_loader.load_cooccurrences_train()

        print("processing cooccurrences numpy array (test)")
        coocs_test = dataset_helper.generate_cooccurrences_array(locations_test)
        file_loader.save_cooccurrences_test(coocs_test)
        coocs_test = file_loader.load_cooccurrences_test()


        print("processing coocs for met in next (train)")
        coocs_met_in_next_train = np.copy(coocs_train)
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] <= second_period_time_bin_max]
        coocs_met_in_next_train = coocs_met_in_next_train[coocs_met_in_next_train[:, 3] > second_period_time_bin_min]

        print("finding met in next people (train)")
        #met_in_next_train = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_train)
        print("saving met in next people (train)")
        file_loader.save_met_in_next_train(coocs_met_in_next_train)

        print("processing coocs for met in next (test)")
        coocs_met_in_next_test = np.copy(coocs_test)
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] <= third_period_time_bin_max]
        coocs_met_in_next_test = coocs_met_in_next_test[coocs_met_in_next_test[:, 3] > third_period_time_bin_min]

        print("finding met in next people (test)")
        #met_in_next_test = predictor.extract_and_remove_duplicate_coocs(coocs_met_in_next_test)
        print("saving met in next people (test)")
        file_loader.save_met_in_next_test(coocs_met_in_next_test)


        print("processing dataset for machine learning (train)")
        X_train, y_train = predictor.generate_dataset(users_train, countries_train, locations_train, coocs_train, coocs_met_in_next_train, first_period_datetime_min, first_period_datetime_max, selected_users)
        print("processing dataset for machine learning (test)")
        X_test, y_test = predictor.generate_dataset(users_test, countries_test, locations_test, coocs_test, coocs_met_in_next_test, second_period_datetime_min, second_period_datetime_max, selected_users)

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
        file_loader.save_x_and_y(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predictor.predict(X_train, y_train, X_test, y_test)
if __name__ == '__main__':
    r = Run(country="Sweden")
    r.update_all_data()
