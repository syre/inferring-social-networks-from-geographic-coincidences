#!/usr/bin/env python3
import json
import requests
import time
import pickle

def fetch_missing_geographical_data():
    """
    Uses reverse geocoding from OpenStreetMap to insert the missing geographical data
    """
    
    URL = "http://nominatim.openstreetmap.org/reverse"
    with open('missing_records.json', 'r') as infile:
            records = json.load(infile)
    addresses = []
    faults = []
    for record in records:
        rec_id = record[0]
        lng = record[1]
        lat = record[2]
        print(lng, lat)
        payload = {"format":"json", "lon":lng, "lat":lat, "email":"syrelyre@gmail.com", "accept-language":"en-us", "User-Agent":"Data Science App"}
        while True:
            try:
                response = json.loads(requests.get(URL, params=payload).text)
                break
            except requests.exceptions.ConnectionError as e:
                print("Exception: {}".format(str(e)))
                time.sleep(1)
        
        if "address" in response:
            try:
                address = response["address"]
                address["lat"] = lat
                address["lng"] = lng
                addresses.append(address)
            except ValueError as e:
                print("ValueError: {}".format(str(e)))
                faults.append((rec_id, lng, lat))
        time.sleep(1)

    with open('missing_data.json', 'w') as outfile:
        json.dump(addresses, outfile)
        
    if faults:
        with open('faults.pickle', 'wb') as outfile:
            pickle.dump(faults, outfile)


if __name__ == '__main__':
	fetch_missing_geographical_data()