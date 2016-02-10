#!/usr/bin/env python3
from collections import defaultdict
import json
import pprint

def save_json_data(data, filename="geodata.geojson"):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def generate_geojson(input_dict):
    geo_dict = defaultdict(dict)

    features = []
    
    count = 0
    for user,_ in wanted_data.items():
        feature = defaultdict(dict)
        properties = defaultdict(dict)
        geometry = defaultdict(dict)
        geometry['type'] = "MultiLineString"
        feature['type'] = "Feature"
        feature['id'] = user
        properties['name'] = 'null'
        properties['times'] = wanted_data[user]['time_start']
        feature['properties'] = properties
        geometry['coordinates'] = [wanted_data[user]['lat_long']]
        feature['geometry'] = geometry
        features.append(feature)
        
    geo_dict['type'] = "FeatureCollection"
    geo_dict['features'] = features
    return geo_dict



raw_data = defaultdict(dict)
wanted_data = defaultdict(dict)


f = "all_201509.json"
with open(f) as json_file:
    raw_data = json.load(json_file)


for data in raw_data:
    if data['country'] == "Japan":
        if data['useruuid'] in wanted_data:
            wanted_data[data['useruuid']]['lat_long'].append([data['longitude'], data['latitude']])
            wanted_data[data['useruuid']]['time_start'].append(data['start_time'])
        else:
            wanted_data[data['useruuid']]['lat_long'] = [[data['longitude'], data['latitude']]]
            wanted_data[data['useruuid']]['time_start'] = [data['start_time']]


geo_json = generate_geojson(wanted_data)
save_json_data(geo_json)


