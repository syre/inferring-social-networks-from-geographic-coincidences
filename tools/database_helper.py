#!/usr/bin/env python3
import psycopg2
import json
import math


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

settings_dict = load_login(file_name="settings.cfg", key_split="=")

conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password='{}'".format(settings_dict["HOSTNAME"], settings_dict["DBNAME"], settings_dict["USER"], settings_dict["PASS"]))

CREATE_TABLE_USER = """CREATE TABLE "user" ( useruuid text PRIMARY KEY NOT NULL)"""
CREATE_TABLE_PLACE = """CREATE TABLE "place" (name text PRIMARY KEY)"""
CREATE_TABLE_AREA = """CREATE TABLE "area" (name text PRIMARY KEY)"""
CREATE_TABLE_COUNTRY = """CREATE TABLE "country" (name text PRIMARY KEY)"""
CREATE_TABLE_REGION = """CREATE TABLE "region" (name text PRIMARY KEY)"""
CREATE_TABLE_LOCATION = """CREATE TABLE "location" ( id SERIAL PRIMARY KEY,
                                                   start_time text NOT NULL,
                                                   end_time text NOT NULL,
                                                   location GEOGRAPHY(POINT,4326),
                                                   altitude INTEGER NOT NULL,
                                                   accuracy INTEGER NOT NULL,
                                                   region text references region(name),
                                                   country text references country(name),
                                                   area text references area(name),
                                                   place text references place(name),
                                                   useruuid text references "user" (useruuid))"""
def db_setup():
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS POSTGIS;")
    cursor.execute(CREATE_TABLE_USER)
    cursor.execute(CREATE_TABLE_PLACE)
    cursor.execute(CREATE_TABLE_AREA)
    cursor.execute(CREATE_TABLE_COUNTRY)
    cursor.execute(CREATE_TABLE_REGION)
    cursor.execute(CREATE_TABLE_LOCATION)
    conn.commit()

def db_teardown():
    cursor = conn.cursor()
    cursor.execute("DROP TABLE user, place, area, country, region, location")

def insert_all_from_json():
    file_names = ["all_201509.json","all_201510.json","all_201511.json"]
    for file_name in file_names:
        with open(file_name) as json_file:
            raw_data = json.load(json_file)
        for count, row in enumerate(raw_data):
            if row["region"]:
                insert_region(row)
            if row["country"]:
                insert_country(row)
            if row["area"]:
                insert_area(row)
            if row["name"]:
                insert_place(row)
            insert_user(row)
            insert_location(row)
            if (count % 100 == 0):
                print(row["useruuid"])

def insert_user(row):
    cursor = conn.cursor()
    cursor.execute("""SELECT * from "user" where "user".useruuid = (%s)""",(row["useruuid"],))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO "user" (useruuid) VALUES (%s)""",(row["useruuid"],))
        conn.commit()

def insert_region(row):
    cursor = conn.cursor()
    cursor.execute("""SELECT * from region where region.name = (%s)""",(row["region"],))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO region (name) VALUES (%s)""",(row["region"],))
        conn.commit()

def insert_location(row):
    cursor = conn.cursor()
    area = None
    country = None
    place = None
    region = None

    if row["area"]:
        area = row["area"]
    if row["country"]:
        country = row["country"]
    if row["name"]:
        place = row["name"]
    if row["region"]:
        region = row["region"]
    cursor.execute(""" INSERT INTO location (start_time, end_time, location, altitude, accuracy, region, country, area, place, useruuid)
                   VALUES (%s,%s,ST_SetSRID(ST_MakePoint(%s, %s),4326),%s,%s,%s,%s,%s,%s,%s)""",(row["start_time"],row["end_time"],row["longitude"],row["latitude"],
                    row["altitude"],row["accuracy"], region, country, area, place, row["useruuid"]))
    conn.commit()

def insert_country(row):
    cursor = conn.cursor()
    cursor.execute("""SELECT * from country where country.name = (%s)""",(row["country"],))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO country (name) VALUES (%s)""",(row["country"],))
        conn.commit()

def insert_area(row):
    cursor = conn.cursor()
    cursor.execute("""SELECT * from area where area.name = (%s)""",(row["area"],))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO area (name) VALUES (%s)""",(row["area"],))
        conn.commit()

def insert_place(row):
    cursor = conn.cursor()
    cursor.execute("""SELECT * from place where place.name = (%s)""",(row["name"],))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO place (name) VALUES (%s)""",(row["name"],))
        conn.commit()

"""
    not using execute(%s and tuple arg) because it doesnt work!!!
"""
def get_distributions(feature, num_bins = 20):
    cursor = conn.cursor()
    cursor.execute("""SELECT max({}) FROM location""".format(feature))
    max_val = int(cursor.fetchone()[0])
    query = "SELECT "+", ".join(["count(CASE WHEN {2} >= {0} AND {2} < {1} THEN 1 END)".format(element,element+(max_val/num_bins), feature) for element in range(0,max_val,int(max_val/num_bins))])+""" from location"""
    
    cursor.execute(query)
    bucketized = [element for element in range(0, max_val, int(max_val/num_bins))]
    results = list(cursor.fetchall()[0])
    return [{"Number":x[0],"Count":x[1]} for x in zip(bucketized, results)]

def drop_tables():
    cursor = conn.cursor()
    cursor.execute("DROP SCHEMA PUBLIC CASCADE")
    cursor.execute("CREATE SCHEMA PUBLIC")
    conn.commit()

if __name__ == '__main__':
    print(get_distributions("accuracy"))