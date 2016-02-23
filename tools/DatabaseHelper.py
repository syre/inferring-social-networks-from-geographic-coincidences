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
        #print(os.listdir())
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

    def get_distributions(self, feature, num_bins = 20):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT max({}) FROM location""".format(feature))
        max_val = int(cursor.fetchone()[0])
        query = "SELECT "+", ".join(["count(CASE WHEN {2} >= {0} AND {2} < {1} THEN 1 END)".format(element,element+(max_val/num_bins), feature) for element in range(0,max_val,int(max_val/num_bins))])+""" from location"""
        
        cursor.execute(query)
        bucketized = [str(element)+"-"+str(element+max_val/num_bins) for element in range(0, max_val, int(max_val/num_bins))]
        results = list(cursor.fetchall()[0])
        return [{"Number":x[0],"Count":x[1]} for x in zip(bucketized, results)]

    def drop_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP SCHEMA PUBLIC CASCADE")
        cursor.execute("CREATE SCHEMA PUBLIC")
        self.conn.commit()

if __name__ == '__main__':
    d = DatabaseHelper()
    d.drop_tables()
    d.db_setup()
    d.insert_all_from_json()