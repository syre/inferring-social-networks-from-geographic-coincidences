#!/usr/bin/env python3
from collections import defaultdict
import json
import pprint
import dateutil
import dateutil.parser
import random

import geojson
from geojson import Feature, FeatureCollection, GeometryCollection, MultiPoint, MultiLineString


def gen_hex_colors(allready_gen=[]):
    r = lambda: random.randint(0,255)
    color = '#%02X%02X%02X' % (r(),r(),r())
    while color in allready_gen:
        r = lambda: random.randint(0,255)
        color = '#%02X%02X%02X' % (r(),r(),r())
    return color

def save_json_data(data, filename="geodata.geojson"):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def generate_geojson_using_objects(input_dict):
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


def generate_geojson(input_dict):
    geo_dict = defaultdict(dict)
    features = []
    for user,_ in wanted_data.items():
        feature = defaultdict(dict)
        style = defaultdict(dict)
        properties = defaultdict(dict)
        geometries = [] #defaultdict(dict)
        geometry = defaultdict(dict)
        geometry['type'] = "MultiLineString"
        geometry_circle = defaultdict(dict)
        geometry_circle['type'] = "MultiPoint"

        feature['type'] = "Feature"
        feature['id'] = user
        style['color'] = wanted_data[user]['color']
        feature['style'] = style
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


        geometry_circle['coordinates'] = multipoints
        properties['name'] = 'null'
        properties['times'] = wanted_data[user]['time_start']
        properties['circles']['opacities'] = opacities


        feature['properties'] = properties
        geometry['coordinates'] = [wanted_data[user]['lat_long']]
        geometries.append(geometry)
        geometries.append(geometry_circle)
        feature['geometries'] = geometries
        features.append(feature)
        
    geo_dict['type'] = "FeatureCollection"
    geo_dict['features'] = features
    return geo_dict


def get_data(country, raw_data):
    wanted_data = defaultdict(dict)
    count_tags = 0
    generated_colors = []
    for data in raw_data:
        if data['country'] == country:
            start_time = dateutil.parser.parse(data['start_time'])
            end_time = dateutil.parser.parse(data['end_time'])
            diff = end_time-start_time
            #print(diff.total_seconds())
            if data['useruuid'] in wanted_data:
                wanted_data[data['useruuid']]['lat_long'].append([data['longitude'], data['latitude']])
                wanted_data[data['useruuid']]['time_start'].append(data['start_time'])
                wanted_data[data['useruuid']]['time_diff'].append(diff.total_seconds())
                wanted_data[data['useruuid']]['total_diff'] += diff.total_seconds()
            else:
                wanted_data[data['useruuid']]['lat_long'] = [[data['longitude'], data['latitude']]]
                wanted_data[data['useruuid']]['time_start'] = [data['start_time']]
                wanted_data[data['useruuid']]['time_diff'] = [diff.total_seconds()]
                wanted_data[data['useruuid']]['total_diff'] = diff.total_seconds()
                color = gen_hex_colors(generated_colors)
                wanted_data[data['useruuid']]['color'] = color
                generated_colors.append(color)
            count_tags += 1
    return wanted_data
raw_data = defaultdict(dict)

f = "all_201509.json"
with open(f) as json_file:
    raw_data = json.load(json_file)

#geo_json = generate_geojson(wanted_data)
#save_json_data(geo_json)

#pprint.pprint(geo_json)

wanted_data = get_data("Japan", raw_data)
geo_json = generate_geojson_using_objects(wanted_data)
validation = geojson.is_valid(geo_json)
if validation['valid'] == 'yes':
    filename = "test_geojson_object_dump.geojson"
    with open(filename, 'w') as outfile:
        geojson.dump(geo_json, outfile)
    #save_json_data()
    print("file saved: {0}".format("test_geojson_object_dump.json"))
else:
    print("geo_json object is not valid: {0}".format(validation['message']))
