#!/usr/bin/env python3
import psycopg2
from collections import defaultdict
import json
import pprint
import dateutil
import dateutil.parser
import random
import os
import geojson
import math
from geojson import Feature, FeatureCollection, GeometryCollection, MultiPoint, MultiLineString

class GeoData(object):
    """docstring for Geo_data"""
    def __init__(self, path_to_settings=""):
        self.settings_dict = self.load_login(file_name="settings.cfg", key_split="=", path=path_to_settings)
        self.conn = None
        self.cursor = None
        self.connect()
        
        
    def check_connection(self):
        if self.cursor is None or self.cursor.closed():
            print("Forbindelsen er lukket!\nÅbner en forbindelse")
            self.connect()

    def close_connection(self):
        self.conn.close()
    def connect(self):
        self.conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password='{}'".
            format(self.settings_dict["HOSTNAME"], self.settings_dict["DBNAME"], 
                self.settings_dict["USER"], self.settings_dict["PASS"]))
        self.cursor = self.conn.cursor()

    def gen_hex_colors(self, allready_gen=[]):
        """Generate a random color in hexadecimal value
        
        Takes a list as input. Generate a random color while the color is in that list. 
        Return the unique color
        
        Keyword Arguments:
            allready_gen {list} -- List of colors which (default: {[]})
        
        Returns:
            [string] -- string representation of the hex color

        Raises:
            NameError -- Raise an exception is the input list is filled, hence there is no more free colors to generate
        """
        
        if len(allready_gen)<(255*255*255):
            r = lambda: random.randint(0,255)
            color = '#%02X%02X%02X' % (r(),r(),r())
            while color in allready_gen:
                r = lambda: random.randint(0,255)
                color = '#%02X%02X%02X' % (r(),r(),r())
            return color
        raise NameError('No more colors left to choose')

    def check_validity(self, data):
        validation = geojson.is_valid(data)
        if validation['valid'] == 'yes':
            return True
        else: 
            print("geo_json object is not valid: {0}".format(validation['message']))
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
        #print("generate_geojson her")
        #print("hej")
        print(len(input_dict.items()))
        #print("hej igen")
        c = 0
        for user,_ in input_dict.items():
            if user.strip() == '' or user is None:
                print("Ingen user??")
            lat_long = input_dict[user]['lat_long']
            #print("lat_long = \n {0}".format(lat_long))
            if not self.validate_lat_long(lat_long):
                print("User {0}, har en fejl i lat_long".format(user))
            #break
            index = 0
            total_diff = input_dict[user]['total_diff']
            multipoints = []
            opacities = []
            #print("h1")
            for diff in input_dict[user]['time_diff']:
                #print("h2")
                multipoints.append(input_dict[user]['lat_long'][index])
                if diff > 0.0:
                    opacities.append(diff/total_diff)
                else:
                    opacities.append(1.0)
                index +=1
            #print("h3")
            geometry_lines = MultiLineString([input_dict[user]['lat_long']])
            geometry_circle = MultiPoint(multipoints)
            #print("h4")
            #print(user)
            #geometries = GeometryCollection([geometry_lines, geometry_circle], properties={'name':'null', 'times':input_dict[user]['time_start'], 'circles': {'opacities': opacities}, 'id': user},
            #    style={'color': input_dict[user]['color']})
            feature_lines = Feature(geometry=geometry_lines, #GeometryCollection([geometry_lines, geometry_circle]), #id=user, 
                properties={'name':'null', 'circles':{'opacities': opacities},'times':input_dict[user]['time_start'], 'id': user}, style={'color': input_dict[user]['color']})
            #feature_circles = Feature(geometry=geometry_circle, #GeometryCollection([geometry_lines, geometry_circle]), #id=user, 
            #    properties={'name':'null', 'times':input_dict[user]['time_start'], 'circles': {'lat_long':input_dict[user]['lat_long'],
            #     'opacities': opacities}, 'id': user}, style={'color': input_dict[user]['color']})
            #print("h5")
            features.append(feature_lines)
            #features.append(feature_circles)
            #if c%10 == 0:
            #print(c)
            c+=1
        return FeatureCollection(features)


    def load_login(self, file_name="login.txt", key_split="##", value_split=",", has_header=False, path=""):
        d = {}
        with open(os.path.join(path, file_name), 'r') as f:
            lines = f.readlines()
        if has_header:
            lines.pop(0)
        for line in lines:
            key_and_values = line.strip().split(key_split)
            key = key_and_values[0]
            values = key_and_values[1].split(value_split)
            d[key] = []
            for value in values:
                d[key].append(value)
            if len(d[key])==1:
                d[key] = d[key][0]
        return d



    def validate_lat_long(self, lat_long_lst):
        for lst in lat_long_lst:
            for lat_long in lst:
                try:
                    float(lat_long)
                    #float(lat_long[1])
                except ValueError:
                    return False
        return True

    def get_geo_data_by_country(self, country, date):
        """Gets useruuid, location, start_time, end_time from the database 
           where country is equal to input parameter.
           Makes a dict where useruuids is the top-key. Generate a hex-color for each user. 
        
        
        Arguments:
            country {string} -- The country which the data should come from
        
        Returns:
            dict -- Dictonary of the collected data
        """
        wanted_data = defaultdict(dict)
        generated_colors = []
        date = dateutil.parser.parse(date)
        # truncate to start of hour
        date = date.replace(minute=0, second=0, microsecond=0)
        self.cursor.execute(""" SELECT useruuid, ST_AsGeoJSON(location) AS geom, start_time, end_time FROM location WHERE country=(%s) AND start_time between (%s) and (%s) + interval '1 day';""", (country,date, date))
        result = self.cursor.fetchall()
        count=0
        for res in result:
            user = res[0]
            lat_long = json.loads(res[1]) #The location if fetched as GeoJSON
            start_time = res[2]
            end_time = res[3]
            diff = end_time-start_time
            if user in wanted_data:
                wanted_data[user]['lat_long'].append(lat_long['coordinates'])
                wanted_data[user]['time_start'].append(res[2])
                wanted_data[user]['time_diff'].append(diff.total_seconds())
                wanted_data[user]['total_diff'] += diff.total_seconds()
            else:
                wanted_data[user]['lat_long'] = [lat_long['coordinates']]
                wanted_data[user]['time_start'] = [res[2]]
                wanted_data[user]['time_diff'] = [diff.total_seconds()]
                wanted_data[user]['total_diff'] = diff.total_seconds()
                color = self.gen_hex_colors(generated_colors)
                wanted_data[user]['color'] = color
                generated_colors.append(color)
            count+=1
        print("Data hentet fra database")
        return wanted_data



    def get_and_generate(self, country, date):
        return self.generate_geojson(self.get_geo_data_by_country(country, date))

if __name__ == '__main__':
    tools_path = "../tools/"
    g = GeoData(tools_path)
    print(g.check_validity(g.get_and_generate(("Japan",))))