#!/usr/bin/env python3
import psycopg2
import json
import math
import os

class DatabaseHelper(object):
    """docstring for DatabaseHelper"""
    def __init__(self, path_to_settings=""):
        self.settings_dict = self.load_login(file_name="settings.cfg", key_split="=", path=path_to_settings)
        self.conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password='{}'".
            format(self.settings_dict["HOSTNAME"], self.settings_dict["DBNAME"], 
                self.settings_dict["USER"], self.settings_dict["PASS"]))

        self.CREATE_TABLE_USER = """CREATE TABLE "user" ( useruuid text PRIMARY KEY NOT NULL)"""
        self.CREATE_TABLE_PLACE = """CREATE TABLE "place" (name text PRIMARY KEY)"""
        self.CREATE_TABLE_AREA = """CREATE TABLE "area" (name text PRIMARY KEY)"""
        self.CREATE_TABLE_COUNTRY = """CREATE TABLE "country" (name text PRIMARY KEY)"""
        self.CREATE_TABLE_REGION = """CREATE TABLE "region" (name text PRIMARY KEY)"""
        self.CREATE_TABLE_LOCATION = """CREATE TABLE "location" ( id SERIAL PRIMARY KEY,
                                                           start_time timestamptz NOT NULL,
                                                           end_time timestamptz NOT NULL,
                                                           location GEOGRAPHY(POINT,4326),
                                                           altitude INTEGER NOT NULL,
                                                           accuracy INTEGER NOT NULL,
                                                           region text references region(name),
                                                           country text references country(name),
                                                           area text references area(name),
                                                           place text references place(name),
                                                           useruuid text references "user" (useruuid))"""
        


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




    def db_setup(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS POSTGIS;")
        cursor.execute(self.CREATE_TABLE_USER)
        cursor.execute(self.CREATE_TABLE_PLACE)
        cursor.execute(self.CREATE_TABLE_AREA)
        cursor.execute(self.CREATE_TABLE_COUNTRY)
        cursor.execute(self.CREATE_TABLE_REGION)
        cursor.execute(self.CREATE_TABLE_LOCATION)
        cursor.execute("CREATE INDEX ON location (start_time)")
        cursor.execute("CREATE INDEX ON location (end_time)")
        cursor.execute("CREATE INDEX ON location using gist (location)")
        cursor.execute("CREATE INDEX ON location (useruuid)")
        self.conn.commit()

    def db_teardown(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE user, place, area, country, region, location")

    def insert_all_from_json(self, path=""):
        file_names = ["all_201509.json","all_201510.json","all_201511.json"]
        for file_name in file_names:
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            print("Antal rows = {}".format(len(raw_data)))
            for count, row in enumerate(raw_data):
                if row["region"]:
                    self.insert_region(row)
                if row["country"]:
                    self.insert_country(row)
                if row["area"]:
                    self.insert_area(row)
                if row["name"]:
                    self.insert_place(row)
                self.insert_user(row)
                self.insert_location(row)
                if (count % 200000 == 0):
                    print(count)

    def insert_user(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT * from "user" where "user".useruuid = (%s)""",(row["useruuid"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO "user" (useruuid) VALUES (%s)""",(row["useruuid"],))
            self.conn.commit()

    def insert_region(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT * from region where region.name = (%s)""",(row["region"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO region (name) VALUES (%s)""",(row["region"],))
            self.conn.commit()

    def insert_location(self, row):
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def insert_country(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT * from country where country.name = (%s)""",(row["country"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO country (name) VALUES (%s)""",(row["country"],))
            self.conn.commit()

    def insert_area(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT * from area where area.name = (%s)""",(row["area"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO area (name) VALUES (%s)""",(row["area"],))
            self.conn.commit()

    def insert_place(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT * from place where place.name = (%s)""",(row["name"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO place (name) VALUES (%s)""",(row["name"],))
            self.conn.commit()

    def get_distributions_numbers(self, feature, num_bins = 20):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT max({}) FROM location""".format(feature))
        max_val = int(cursor.fetchone()[0])
        query = "SELECT "+", ".join(["count(CASE WHEN {2} >= {0} AND {2} < {1} THEN 1 END)".format(element,element+(max_val/num_bins), feature) for element in range(0,max_val,int(max_val/num_bins))])+""" from location"""
        cursor.execute(query)
        bucketized = [str(element)+"-"+str(element+max_val/num_bins) for element in range(0, max_val, int(max_val/num_bins))]
        results = list(cursor.fetchall()[0])
        return [{"Number":x[0],"Count":x[1]} for x in zip(bucketized, results)], {'x_axis': "Number", 'y_axis': "Count"}

    def get_distributions_text(self, feature, num_bins = 20):
        if feature == "country":
            cursor = self.conn.cursor()
            cursor.execute("""select country, count(*) from location group by country order by count(*) desc;""")
            result = cursor.fetchall()
            countries = [row[0] for row in result]
            count = [row[1] for row in result]
            return [{"Countries": x[0], "Count": x[1]} for x in zip(countries, count)], {'x_axis': "Countries", 'y_axis': "Count"}


    def get_locations_for_user(self, useruuid):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT useruuid, start_time, end_time, ST_X(location::geometry), ST_Y(location::geometry) from location where location.useruuid = (%s) """,(useruuid,))
        if cursor.rowcount == 0:
            print("no locations for useruuid")
            return
        else:
            locations = cursor.fetchall()
        return locations
    
    def find_cooccurrences(self, useruuid, cell_size, time_threshold_in_hours):
        cursor = self.conn.cursor()
        locations = self.get_locations_for_user(useruuid)

        cooccurrences = []
        for location in locations:
            start_time = location[1]
            end_time = location[2]
            longitude = location[3]
            latitude = location[4]

            # find coocurrences by taking time_treshold_in_hours/2 before start_time and time_threshold_in_hours/2 after end_time
            # this also means time window can get really long, what are the consequences?
            cursor.execute(""" SELECT useruuid, start_time, end_time, ST_AsGeoJSON(location) AS geom from location where location.useruuid != (%s)
             and (start_time between (%s) - interval '%s minutes' and (%s)) and (end_time between (%s) and (%s) + interval '%s minutes') and abs(ST_X(location::geometry)-(%s)) <= (%s) and abs(ST_Y(location::geometry)-(%s)) <= (%s)""",
             (useruuid, start_time, time_threshold_in_hours/2, start_time, end_time, end_time, time_threshold_in_hours/2, longitude, cell_size, latitude, cell_size))
            
            result = cursor.fetchall()
            if result:
                cooccurrences.extend(result)
        return cooccurrences

    def get_all_users(self):
        cursor = self.conn.cursor()
        cursor.execute(""" SELECT DISTINCT "useruuid" FROM public.user;""")
        return cursor.fetchall()

    def get_locations_by_country(self, country, start_datetime, end_datetime):
        cursor = self.conn.cursor()
        cursor.execute(""" SELECT useruuid, ST_AsGeoJSON(location) AS geom, start_time, end_time FROM location 
            WHERE country=(%s) AND ((start_time, end_time) OVERLAPS ((%s), (%s)));""", (country, start_datetime, end_datetime))
        return cursor.fetchall()

    def drop_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("drop schema public cascade;")
        cursor.execute("create schema public;")
        self.conn.commit()

if __name__ == '__main__':
    d = DatabaseHelper()
    print(d.find_cooccurrences("c98f46b9-43fd-4536-afa0-9b789300fe7a", 0.001, 60*24))
    #d.drop_tables()
    #d.db_setup()
    #d.insert_all_from_json()
