#!/usr/bin/env python3
import psycopg2
import json

HOSTNAME="localhost"
DBNAME="dtudatascience2016"
USER="admin"
PASS="adminpass"
conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password='{}'".format(HOSTNAME, DBNAME, USER, PASS))

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

def drop_tables():
    cursor = conn.cursor()
    cursor.execute("DROP SCHEMA PUBLIC CASCADE")
    cursor.execute("CREATE SCHEMA PUBLIC")
    conn.commit()
    
if __name__ == '__main__':
    #db_setup()
    insert_all_from_json()