#!/usr/bin/env python3
from collections import defaultdict
import json
import pprint
import dateutil
import dateutil.parser
import random


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

def generate_geojson(input_dict):
    geo_dict = defaultdict(dict)

    features = []
    
    count = 0
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
            #print(wanted_data[user]['lat_long'][index])
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



raw_data = defaultdict(dict)
wanted_data = defaultdict(dict)


f = "all_201509.json"
with open(f) as json_file:
    raw_data = json.load(json_file)

count_tags = 0
generated_colors = []
for data in raw_data:
    if data['country'] == "Japan":
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



geo_json = generate_geojson(wanted_data)
save_json_data(geo_json)

#pprint.pprint(geo_json)


