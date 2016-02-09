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
    
    
    for count, t in enumerate(wanted_data.items()): #(user, timestamps)
        feature = defaultdict(dict)
        properties = defaultdict(dict)
        geometry = defaultdict(dict)
        geometry['type'] = "MultiLineString"

        user, lat_long = t
        feature['type'] = "Feature"
        feature['id'] = user
        properties['name'] = 'null'
        feature['properties'] = properties
        geometry['coordinates'] = [lat_long]
        feature['geometry'] = geometry
        features.append(feature)
        if count >=3:
            break
        
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
            wanted_data[data['useruuid']].append([data['longitude'], data['latitude']])
        else:
            wanted_data[data['useruuid']] = [[data['longitude'], data['latitude']]]



geo_json = generate_geojson(wanted_data)
save_json_data(geo_json)
#pprint.pprint(geo_json)
# for count, t in enumerate(wanted_data.items()): #(user, timestamps)
#     user, timestamps = t
#     if count >=7:
#         break
#     print("{0} = {1}".format(user, timestamps))


