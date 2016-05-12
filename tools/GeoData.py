#!/usr/bin/env python3
import json
import dateutil
import dateutil.parser
import geojson
from geojson import Feature, FeatureCollection, MultiPoint, MultiLineString, Point
import DatabaseHelper
from collections import defaultdict
from itertools import combinations


class GeoData(object):

    """docstring for Geo_data"""

    def __init__(self, path_to_settings=""):
        self.databasehelper = DatabaseHelper.DatabaseHelper(path_to_settings)

    def check_validity(self, data):
        validation = geojson.is_valid(data)
        if validation['valid'] == 'yes':
            return True
        else:
            print(
                "geo_json object is not valid: {0}".format(validation['message']))
            return False

    def save_json_data_if_valid(self, data, filename="geodata.geojson"):
        if self.check_validity(data):
            with open(filename, 'w') as outfile:
                geojson.dump(geo_json, outfile)
            print("file saved: {0}".format(filename))
        else:
            print("Something went wrong")

    def generate_geojson(self, input_dict):
        """Generate the geojson dictonary for the geo data

        Arguments:
            input_dict {dict} -- ["raw" dict of the data]

        Returns:
            geojson dict -- geojson dictonary of the "raw" data
        """
        features = []
        print(len(input_dict.items()))
        c = 0
        for user, _ in input_dict.items():
            if user.strip() == '' or user is None:
                print("no user found")
            lat_long = input_dict[user]['lat_long']
            if not self.validate_lat_long(lat_long):
                print("User {0} has an error in lat_long".format(user))
            index = 0
            total_diff = input_dict[user]['total_diff']
            multipoints = []
            opacities = []
            for diff in input_dict[user]['time_diff']:
                multipoints.append(input_dict[user]['lat_long'][index])
                if diff > 0.0:
                    opacities.append(diff/total_diff)
                else:
                    opacities.append(1.0)
                index += 1
            geometry_lines = MultiLineString([input_dict[user]['lat_long']])
            geometry_circle = MultiPoint(multipoints)
            feature_lines = Feature(geometry=geometry_lines,
                                    properties={'name': 'null', 'circles': {'opacities': opacities}, 'times': input_dict[user]['start_time'], 'id': user}, style={'color': input_dict[user]["color"]})
            features.append(feature_lines)
            c += 1
        return FeatureCollection(features)

    def validate_lat_long(self, lat_long_lst):
        for lst in lat_long_lst:
            for lat_long in lst:
                try:
                    float(lat_long)
                except ValueError:
                    return False
        return True

    def get_geo_data_by_country(self, country, start_datetime, end_datetime):
        """Gets useruuid, location, start_time, end_time from the database 
           where country is equal to input parameter.
           Makes a dict where useruuids is the top-key. Generate a hex-color for each user. 


        Arguments:
            country {string} -- The country which the data should come from

        Returns:
            dict -- Dictonary of the collected data
        """
        wanted_data = defaultdict(dict)
        user_colors = self.databasehelper.get_user_colors()
        start_datetime = dateutil.parser.parse(start_datetime)
        end_datetime = dateutil.parser.parse(end_datetime)

        result = self.databasehelper.get_locations_by_country(
            country, start_datetime, end_datetime)
        for res in result:
            user = res[0]
            lat_long = json.loads(res[1])  # The location is fetched as GeoJSON
            start_time = res[2]
            end_time = res[3]
            diff = end_time-start_time
            if user in wanted_data:
                wanted_data[user]['lat_long'].append(lat_long['coordinates'])
                wanted_data[user]['start_time'].append(res[2])
                wanted_data[user]['time_diff'].append(diff.total_seconds())
                wanted_data[user]['total_diff'] += diff.total_seconds()
            else:
                wanted_data[user]['lat_long'] = [lat_long['coordinates']]
                wanted_data[user]['start_time'] = [res[2]]
                wanted_data[user]['time_diff'] = [diff.total_seconds()]
                wanted_data[user]['total_diff'] = diff.total_seconds()
                wanted_data[user]['color'] = user_colors[user]
        print("geodata fetched from database")
        return wanted_data

    def get_geo_data_from_occurrences(self, useruuid1, useruuid2, points_w_distances=[]):
        user_colors = self.databasehelper.get_user_colors()
        # receive locations for user we want co-occurrences on
        locations = self.databasehelper.get_locations_for_user(useruuid1)
        # retrieve locations that co-occur with useruuids locations
        if useruuid2:
            cooccurrences = self.databasehelper.find_cooccurrences(
            useruuid1, points_w_distances, useruuid2=useruuid2)
        else:
            cooccurrences = self.databasehelper.find_cooccurrences(
                            useruuid1, points_w_distances)

        features = []
        # append main user
        features.append(Feature(geometry=MultiLineString([[(loc[3], loc[4]) for loc in locations]]), properties={
                        "id1": useruuid1, "name": "null"}, style={'color': "red"}))
        # append cooccurrences
        for cooccurrence in cooccurrences:
            lat_long = json.loads(cooccurrence[5])
            features.append(Feature(geometry=Point(lat_long["coordinates"]), properties={
                            "id1": useruuid1, "name": "null"}, style={'color': user_colors[useruuid1]}))

        return FeatureCollection(features)

    def get_geo_data_from_all_cooccurrences(self):
        locations = self.databasehelper.get_locations_by_country_only("Sweden")
        features = [Feature(geometry=Point(json.loads(loc[3])["coordinates"])) for loc in locations]
        return FeatureCollection(features)

    def get_and_generate(self, country, start_date, end_date):
        return self.generate_geojson(self.get_geo_data_by_country(country, start_date, end_date))

if __name__ == '__main__':
    tools_path = "../tools/"
    g = GeoData(tools_path)
    print(g.get_geo_data_from_occurrences(
        "c98f46b9-43fd-4536-afa0-9b789300fe7a", 0.001, 60*24))
