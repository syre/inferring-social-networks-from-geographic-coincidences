#!/usr/bin/env python3
import psycopg2
from collections import defaultdict
import json
import pprint
import dateutil
import dateutil.parser
import random

import geojson
from geojson import Feature, FeatureCollection, GeometryCollection, MultiPoint, MultiLineString


def gen_hex_colors(allready_gen=[]):
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


def save_json_data(data, filename="geodata.geojson"):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def generate_geojson(input_dict):
    """Generate the geojson dictonary for the geo data
    
    Arguments:
        input_dict {dict} -- ["raw" dict of the data]
    
    Returns:
        geojson dict -- geojson dictonary of the "raw" data
    """
    features = []
    for user,_ in wanted_data.items():
        geometries = []
        index = 0
        total_diff = wanted_data[user]['total_diff']
        multipoints = []
        opacities = []
        for diff in wanted_data[user]['time_diff']:
            multipoints.append([wanted_data[user]['lat_long'][index]])
            if diff > 0.0:
                opacities.append(diff/total_diff)
            else:
                opacities.append(1.0)
            index +=1

        geometry_lines = MultiLineString([wanted_data[user]['lat_long']])
        geometry_circle = MultiPoint(multipoints)
        geometries.append(geometry_lines)
        geometries.append(geometry_circle)
        feature = Feature(geometries=geometries, id=user, 
            properties={'name':'null', 'times':wanted_data[user]['time_start'], 'circles': {'opacities': opacities}},
            style={'color': wanted_data[user]['color']})
        features.append(feature)
    return FeatureCollection(features)

# def get_data(country, raw_data):
#     wanted_data = defaultdict(dict)
#     count_tags = 0
#     generated_colors = []
#     for data in raw_data:
#         if data['country'] == country:
#             start_time = dateutil.parser.parse(data['start_time'])
#             end_time = dateutil.parser.parse(data['end_time'])
#             diff = end_time-start_time
#             #print(diff.total_seconds())
#             if data['useruuid'] in wanted_data:
#                 wanted_data[data['useruuid']]['lat_long'].append([data['longitude'], data['latitude']])
#                 wanted_data[data['useruuid']]['time_start'].append(data['start_time'])
#                 wanted_data[data['useruuid']]['time_diff'].append(diff.total_seconds())
#                 wanted_data[data['useruuid']]['total_diff'] += diff.total_seconds()
#             else:
#                 wanted_data[data['useruuid']]['lat_long'] = [[data['longitude'], data['latitude']]]
#                 wanted_data[data['useruuid']]['time_start'] = [data['start_time']]
#                 wanted_data[data['useruuid']]['time_diff'] = [diff.total_seconds()]
#                 wanted_data[data['useruuid']]['total_diff'] = diff.total_seconds()
#                 color = gen_hex_colors(generated_colors)
#                 wanted_data[data['useruuid']]['color'] = color
#                 generated_colors.append(color)
#             count_tags += 1
#     return wanted_data

def load_login(file_name="login.txt", key_split="##", value_split=",", has_header=False):
    d = {}
    with open(file_name, "r") as f:
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


def get_data_database(country):
    settings_dict = load_login(file_name="settings.cfg", key_split="=")
    conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password='{}'".format(settings_dict["HOSTNAME"], settings_dict["DBNAME"], settings_dict["USER"], settings_dict["PASS"]))

    wanted_data = defaultdict(dict)
    count_tags = 0
    generated_colors = []
    cursor = conn.cursor()
    cursor.execute(""" SELECT useruuid, ST_AsGeoJSON(location) AS geom, start_time, end_time FROM location WHERE country=(%s);""", (country,))
    result = cursor.fetchall()
    for res in result:
        user = res[0]
        lat_long = json.loads(res[1])
        #print(lat_long['coordinates'])
        
        start_time = dateutil.parser.parse(res[2])
        end_time = dateutil.parser.parse(res[3])
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
            color = gen_hex_colors(generated_colors)
            wanted_data[user]['color'] = color
            generated_colors.append(color)

    return wanted_data

wanted_data = get_data_database("Japan")
# wanted_data = get_data("Japan", raw_data)
geo_json = generate_geojson(wanted_data)
validation = geojson.is_valid(geo_json)
if validation['valid'] == 'yes':
    filename = "test_geojson_object_dump.geojson"
    with open(filename, 'w') as outfile:
        geojson.dump(geo_json, outfile)
    #save_json_data()
    print("file saved: {0}".format(filename))
else:
    print("geo_json object is not valid: {0}".format(validation['message']))
