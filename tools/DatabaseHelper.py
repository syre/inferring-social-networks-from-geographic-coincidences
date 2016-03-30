#!/usr/bin/env python3
from collections import defaultdict
import psycopg2
import json
import math
import os
from collections import defaultdict, Counter
import random
import datetime
import dateutil
from dateutil import parser
import time
from tqdm import tqdm

from GeoCalculation import GeoCalculation

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
                                                           useruuid text references "user" (useruuid),
                                                           spatial_loc_id integer references "spatial_location" (id))"""

        self.CREATE_TABLE_SPATIAL_LOCATION = """CREATE TABLE "spatial_location" (
                                                    id SERIAL PRIMARY KEY,
                                                    lng_twodec NUMERIC(5,2) NOT NULL,
                                                    lat_twodec NUMERIC(5,2) NOT NULL)"""


        self.CREATE_TABLE_TIME_BIN = """CREATE TABLE "time_bin" (
                                                    id SERIAL PRIMARY KEY,
                                                    time_bin_number integer NOT NULL,
                                                    loc_id integer references "location" (id))"""

        self.geo_calc = GeoCalculation()
        # if database is setup
        if self.db_setup_test():
            all_users = self.get_distinct_feature("useruuid","user")
            colors = []
            self.user_colors = defaultdict(dict)
            for user in all_users:
                color = self.gen_hex_colors(colors)
                self.user_colors[user] = color
                colors.append(color)
    
    def get_user_colors(self):
        return self.user_colors

    def gen_hex_colors(self, allready_gen=[]):
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


    def db_setup_test(self):
        cursor = self.conn.cursor()
        cursor.execute("select exists(select * from information_schema.tables where table_name=%s)", ('user',))
        return cursor.fetchone()[0]

    def db_setup(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS POSTGIS;")
        cursor.execute(self.CREATE_TABLE_USER)
        cursor.execute(self.CREATE_TABLE_PLACE)
        cursor.execute(self.CREATE_TABLE_AREA)
        cursor.execute(self.CREATE_TABLE_COUNTRY)
        cursor.execute(self.CREATE_TABLE_REGION)
        cursor.execute(self.CREATE_TABLE_SPATIAL_LOCATION)
        cursor.execute(self.CREATE_TABLE_LOCATION)
        cursor.execute(self.CREATE_TABLE_TIME_BIN)
        self.conn.commit()

    def db_create_indexes(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX ON location (start_time)")
        cursor.execute("CREATE INDEX ON location (end_time)")
        cursor.execute("CREATE INDEX ON location using gist (location)")
        cursor.execute("CREATE INDEX ON location (useruuid)")
        cursor.execute("CREATE INDEX user_loc_index ON location (useruuid,location)")
        cursor.execute("CREATE INDEX country_name_index ON country (name)")
        cursor.execute("CREATE INDEX spatial_location_lng_twodec_index ON spatial_location (lng_twodec)")
        cursor.execute("CREATE INDEX spatial_location_lat_twodec_index ON spatial_location (lat_twodec)")
        cursor.execute("CREATE INDEX ON time_bin(time_bin_number)")
        self.conn.commit()


    def db_teardown(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE user, place, area, country, region, location, spatial_location")

    def insert_all_from_json(self, path=""):
        file_names = ["all_201509.json","all_201510.json","all_201511.json"]
        for file_name in tqdm(file_names):
            with open(os.path.join(path, file_name), 'r') as json_file:
                raw_data = json.load(json_file)
            for row in tqdm(raw_data, nested=True):
                if row["region"]:
                    self.insert_region(row)
                if row["country"]:
                    self.insert_country(row)
                if row["area"]:
                    self.insert_area(row)
                if row["name"]:
                    self.insert_place(row)
                self.insert_user(row)
                self.insert_spatial_location(row)
                self.insert_location(row)

    def insert_user(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT 1 from "user" where "user".useruuid = (%s) limit 1""",(row["useruuid"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO "user" (useruuid) VALUES (%s)""",(row["useruuid"],))
            self.conn.commit()


    def insert_region(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT 1 from region where region.name = (%s) limit 1""",(row["region"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO region (name) VALUES (%s)""",(row["region"],))
            self.conn.commit()

    def insert_spatial_location(self, row):
        cursor = self.conn.cursor()
        lng_twodec = int(row["longitude"] * 10**2) / 10.0**2
        lat_twodec = int(row["latitude"] * 10**2) / 10.0**2
        
        cursor.execute(""" SELECT 1 from spatial_location where spatial_location.lng_twodec = (%s) and spatial_location.lat_twodec = (%s) limit 1""", (lng_twodec, lat_twodec))
        if cursor.rowcount == 0:
            cursor.execute(""" INSERT INTO spatial_location (lng_twodec, lat_twodec) values (%s,%s)""", (lng_twodec, lat_twodec))
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
        lng_twodec = int(row["longitude"] * 10**2) / 10.0**2
        lat_twodec = int(row["latitude"] * 10**2) / 10.0**2
        cursor.execute(""" SELECT id from spatial_location where spatial_location.lng_twodec = (%s) and spatial_location.lat_twodec = (%s)""", (lng_twodec, lat_twodec))
        spatial_id = cursor.fetchone()[0]
        cursor.execute(""" INSERT INTO location (start_time, end_time, location, altitude, accuracy, region, country, area, place, useruuid, spatial_loc_id)
                       VALUES (%s,%s,ST_SetSRID(ST_MakePoint(%s, %s),4326),%s,%s,%s,%s,%s,%s,%s, %s) RETURNING id""",(row["start_time"],row["end_time"],row["longitude"],row["latitude"],
                        row["altitude"],row["accuracy"], region, country, area, place, row["useruuid"], spatial_id))
        loc_id = cursor.fetchone()[0]
        self.insert_timebin(row["start_time"], row["end_time"], loc_id)
        self.conn.commit()


    def insert_timebin(self, start_time, end_time, loc_id):
        cursor = self.conn.cursor()
        start_time = parser.parse(start_time)
        end_time = parser.parse(end_time)
        duration = end_time-start_time
        min_datetime = parser.parse('2015-08-09 00:00:00+02')
        duration = duration.total_seconds()/60.0 #in minutes
        start_diff = (start_time-min_datetime).total_seconds()/60.0
        start_bin = math.floor(start_diff/60) #tag højde for 0??
        end_bin = math.ceil((duration/60))
        time_bins = list(range(start_bin, start_bin+end_bin+1))
        for tbin in time_bins:
            cursor.execute("""INSERT INTO time_bin (time_bin_number, loc_id) VALUES(%s, %s) """,(tbin,loc_id))

    def insert_country(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT 1 from country where country.name = (%s) limit 1""",(row["country"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO country (name) VALUES (%s)""",(row["country"],))
            self.conn.commit()

    def insert_area(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT 1 from area where area.name = (%s) limit 1""",(row["area"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO area (name) VALUES (%s)""",(row["area"],))
            self.conn.commit()

    def insert_place(self, row):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT 1 from place where place.name = (%s) limit 1""",(row["name"],))
        if cursor.rowcount == 0:
            cursor.execute("""INSERT INTO place (name) VALUES (%s)""",(row["name"],))
            self.conn.commit()

    def get_distributions_numbers(self, feature, num_bins = 20, max_value=0):
        cursor = self.conn.cursor()
        if not max_value:
            cursor.execute("""SELECT max({}) FROM location""".format(feature))
            max_value = int(cursor.fetchone()[0])

        step_size = int(max_value/num_bins)
        end_value = max_value-int(max_value/num_bins)

        query = "SELECT "+", ".join(["count(CASE WHEN {2} >= {0} AND {2} < {1} THEN 1 END)".
                              format(element,element+step_size, feature) for element in range(0,end_value,step_size)])+", count(CASE WHEN {0} > {1} THEN 1 END)".format(feature, max_value-step_size)+" from location"
        cursor.execute(query)
        results = list(cursor.fetchall()[0])

        bucketized = [str(element)+"-"+str(element+step_size) for element in range(0, end_value, step_size)]
        bucketized.extend([str(max_value-step_size)+"<"])

        return {"results":[{"Number":x[0],"Count":x[1], "Order":index} for index,x in enumerate(zip(bucketized, results))], 'x_axis': "Number", 'y_axis': "Count"}

    def get_distributions_categories(self, feature, num_bins = 20):
        if feature == "country":
            cursor = self.conn.cursor()
            cursor.execute("""select country, count(*) from location group by country order by count(*) desc;""")
            result = cursor.fetchall()
            countries = [row[0] for row in result]
            count = [row[1] for row in result]
            # return with order value =(-count) to sort by count descending
            return {"results":[{"Countries": x[0], "Count": x[1], "Order":-x[1]} for x in zip(countries, count)], 'x_axis': "Countries", 'y_axis': "Count"}

    def auxiliary_function_velocity(self, lst_start_time, lst_end_time, lst_points):
        previous_point = lst_points[0]
        previous_start_time = lst_start_time[0]
        previous_end_time = lst_end_time[0]

        count = 0
        total_km_hour = 0.0
        for index, end_time in enumerate(lst_end_time[1:], start=1):
            duration = end_time - previous_start_time 
            duration = duration.total_seconds()
            if duration > 0.0:
                distance = self.geo_calc.distance_between(lst_points[index], previous_point)
                meter_pr_second = (distance/duration)
                km_hour = ((meter_pr_second*18)/5)
                count += 1
                total_km_hour += km_hour
            previous_point = lst_points[index]
            previous_start_time = lst_start_time[index]
            previous_end_time = lst_end_time[index]
        if count > 0:
            return (total_km_hour/count)
        else:
            print("hov!!!")
            return 0.0

    def get_velocity_for_users(self, country):
        raw_data = defaultdict(dict)
        cursor = self.conn.cursor()
        cursor.execute("""select useruuid,  start_time, end_time,  ST_X(location::geometry), ST_Y(location::geometry) from location where country=(%s) order by start_time;""",(country,))
        result = cursor.fetchall()

        for row in result:
            if row[0] not in raw_data:
                raw_data[row[0]]['start_time'] = [row[1]]
                raw_data[row[0]]['end_time'] = [row[2]]
                raw_data[row[0]]['lat_long'] = [(row[4],row[3])]
            else:
                raw_data[row[0]]['start_time'].append(row[1])
                raw_data[row[0]]['end_time'].append(row[2])
                raw_data[row[0]]['lat_long'].append((row[4],row[3]))

        data = []
        names = []
        for user in raw_data:
            data.append(self.auxiliary_function_velocity(raw_data[user]['start_time'], raw_data[user]['end_time'], raw_data[user]['lat_long']))
            names.append(user)

        data, names = zip(*sorted(zip(data, names)))
        return data, names

    def get_locations_for_user(self, useruuid, country=None):
        country_query =""
        if country:
            country_query = " AND location.country = '{}'".format(country)
        cursor = self.conn.cursor()
        cursor.execute("""SELECT useruuid, start_time, end_time, ST_X(location::geometry), ST_Y(location::geometry) from location 
            where location.useruuid = (%s) """+country_query+""";""",(useruuid,))
        if cursor.rowcount == 0:
            print("no locations for useruuid")
            return
        else:
            locations = cursor.fetchall()
        return locations

    def find_cooccurrences(self, useruuid, points_w_distances=[], useruuid2=None, asGeoJSON=True):
        """ find all cooccurrences for a user
        
        find all cooccurrences for a user within a cell_size and time window (time_threshold_in_minutes)
        Can also filter cooccurences by geometry point (see points_w_distances). Here it finds only cooccurences which is outside of these points
        based on the given distance 
        
        Arguments:
            useruuid {string} -- ID of user
            cell_size {float} -- Size of the cell
            time_threshold_in_minutes {integer} -- [description]
        
        Keyword Arguments:
            points_w_distances {list of lists} -- list of lists with point (longitude,latitude) (as tuple) and distance in meter (default: {[]})
                                                  Example [[(139.743862,35.630338), 1000]]
            useruuid2 {string} -- optional argument for when you only want cooccurrences with that other user
        
        Returns:
            list -- list cooccurrences with useruuid, start_time, end_time, geojsoned location
        """
        cursor = self.conn.cursor()
        
        start = query = ""
        
        if points_w_distances:
            start ="AND NOT ST_DWithin(location, ST_MakePoint("
            query = " AND NOT ST_DWithin(location, ST_MakePoint(".join(["{0}, {1}), {2})". format(element[0][0],element[0][1],element[1]) for element in points_w_distances])

        
        second_user_query = ""
        if useruuid2:
            second_user_query = " AND location.useruuid = '{}' ".format(useruuid2)
        if asGeoJSON:
            lat_lng_format = "ST_AsGeoJSON(location)"
        else:
            lat_lng_format = "ST_X(location::geometry), ST_Y(location::geometry)"

        cursor.execute("""WITH aux_user_table 
     AS (SELECT location.id, 
                useruuid                    AS USER, 
                t.arr                       AS aux_timebins, 
                spatial_location.lng_twodec AS aux_spatial_lng, 
                spatial_location.lat_twodec AS aux_spatial_lat
         FROM   location 
                inner join spatial_location 
                        ON spatial_location.id = location.spatial_loc_id 
                left join (SELECT loc_id, 
                                  Array_agg(time_bin_number) AS arr 
                           FROM   time_bin 
                           GROUP  BY time_bin.loc_id) t 
                       ON t.loc_id = location.id 
         WHERE  location.useruuid = 'f67ae795-1f2b-423c-ba30-cdd5cbb23662') 
SELECT DISTINCT ON (location.id) 
                                 useruuid,
                                 """ + lat_lng_format + """,
                                 u.arr, 
                                 aux_user_table.aux_timebins 
FROM   location 
       inner join spatial_location 
               ON spatial_location.id = location.spatial_loc_id 
       left join (SELECT loc_id, 
                         Array_agg(time_bin_number) AS arr 
                  FROM   time_bin 
                  GROUP  BY time_bin.loc_id) u 
              ON u.loc_id = location.id 
       inner join aux_user_table 
               ON location.useruuid != aux_user_table.USER 
WHERE  aux_user_table.aux_timebins && u.arr 
       AND spatial_location.lng_twodec = aux_spatial_lng 
       AND spatial_location.lat_twodec = aux_spatial_lat
       """ + second_user_query + (start + query) + "ORDER BY location.id" + ";",(useruuid))
        
        return cursor.fetchall()



    def find_cooccurrences_within_area(self, lng, lat, time_bin):
        cursor = self.conn.cursor()
        cursor.execute("""
                SELECT DISTINCT(useruuid)
                FROM location
                INNER JOIN spatial_location ON location.spatial_loc_id=spatial_location.id
                INNER JOIN time_bin on location.id=time_bin.loc_id
                WHERE lng_twodec=(%s) AND lat_twodec=(%s)
                AND time_bin.time_bin_number = (%s)
            """,(lng, lat, time_bin))

        return [row[0] for row in cursor.fetchall()]


    def get_distribution_cooccurrences(self, x_useruuid, y_useruuid, time_threshold_in_minutes=60*24, cell_size=0.001):

        cooccurrences = self.find_cooccurrences(x_useruuid, cell_size, time_threshold_in_minutes, points_w_distances=[], useruuid2=y_useruuid)

        time_dict = {}
        start = datetime.datetime.strptime("01-09-2015", "%d-%m-%Y")
        end = datetime.datetime.strptime("01-12-2015", "%d-%m-%Y")
        # generate range of dates from start to end with interval of 1 day
        date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

        for date in date_generated:
                time_dict[date.strftime("%d/%m/%Y")] = 0
        for cocc in cooccurrences:
            start_date = cocc[1].strftime("%d/%m/%Y")
            end_date = cocc[2].strftime("%d/%m/%Y")
            time_dict[start_date] += 1

        return [{"Date":date_string, "Cooccurrences":value} for date_string,value in time_dict.items()]
        
    def dump_missing_geographical_rows(self):
        cursor = self.conn.cursor()
        query = "select id, ST_X(location::geometry), ST_Y(location::geometry) from location where coalesce(location.country, '') = '';"
        cursor.execute(query)
        records = cursor.fetchall()
        list_of_records = []
        for record in records:
            list_of_records.append(record)

        with open('missing_records.json', 'w') as outfile:
            json.dump(list_of_records, outfile)

    def insert_missing_geographical_data(self):
        with open('missing_data.json', 'r') as infile:
            records = json.load(infile)

        for record in records:
            country = address["country"]
            if state in address:
                area = address["state"]
            elif state_district in address:
                area = address["state_district"]
            else:
                area = ""
                print("no area found")

            if "city" in address:
                place = address["city"]
            elif "town" in address:
                place = address["town"]
            else:
                place = ""
                print("no place found")

            cursor.execute("UPDATE location set country = (%s), area= (%s), place= (%s) ", (country, area, place))
        self.conn.commit()

    def get_all_cooccurrences_as_network(self, time_threshold_in_minutes=60*24, cell_size=0.001):
        cursor = self.conn.cursor()

        distinct_users = self.get_distinct_feature("useruuid", "user")
        nodes = []
        edges = []
        count = 0
        for user in distinct_users:
            locations = self.get_locations_for_user(user)
            if not any(node['id'] == user for node in nodes):
                nodes.append({"id":user, "label":user, "color":self.user_colors[user]})
            cooccurrences = []
            for location in locations:
                start_time = location[1]
                end_time = location[2]
                longitude = location[3]
                latitude = location[4]

            # find coocurrences by taking time_treshold_in_hours/2 before start_time and time_threshold_in_hours/2 after end_time
            # this also means time window can get really long, what are the consequences?
            cursor.execute(""" SELECT useruuid AS geom from location where location.useruuid != (%s)
             and (start_time between (%s) - interval '%s minutes' and (%s)) and (end_time between (%s) and (%s) + interval '%s minutes') and abs(ST_X(location::geometry)-(%s)) <= (%s) and abs(ST_Y(location::geometry)-(%s)) <= (%s)""",
             (user, start_time, time_threshold_in_minutes/2, start_time, end_time, end_time, time_threshold_in_minutes/2, longitude, cell_size, latitude, cell_size))
            
            result = cursor.fetchall()
            if result:
                cooccurrences.extend(result)
            for cooccurrence in cooccurrences:
                if not any(node['id'] == cooccurrence[0] for node in nodes):
                    nodes.append({"id":cooccurrence[0], "label":cooccurrence[0], "color":self.user_colors[cooccurrence[0]]})
            edges.extend([{"source":user, "target":element[0], "id":index} for index,element in enumerate(cooccurrences, start=count)])
            count += len(cooccurrences)

        return {"nodes":nodes, "edges":edges}

    def get_users_with_most_updates(self):
    	cursor = self.conn.cursor()
    	cursor.execute("select useruuid from location group by useruuid order by count(*) desc;")
    	return [element[0] for element in cursor.fetchall()]

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


    def get_boxplot_duration(self, country, for_all_countries=False):
        cursor = self.conn.cursor()
        data = []
        names = []
        if for_all_countries:
            print("hurtig")
            total_rows = []
            cursor.execute("""SELECT country, SUM((end_time - start_time)) AS total_diff_time, count(*) AS number_rows_for_user FROM location GROUP BY country ORDER BY country;""")
            result = cursor.fetchall()
            for row in result:
                country = row[0]
                names.append(country)
                if country != "" and country != " " and country is not None:
                    time = row[1]
                    number_rows_for_country = row[2]

                    total_rows.append(number_rows_for_country)
                    time = time.total_seconds()
                    average_time = "{0:.2f}".format(time/number_rows_for_country)
                    data.append(float(average_time))
                else:
                    print("Tom streng!")
            print("total_rows:")
            print(total_rows)
        else:
            cursor = self.conn.cursor()
            cursor.execute(""" SELECT useruuid, SUM((end_time - start_time)) AS total_diff_time, count(*) AS number_rows_for_user FROM location WHERE country=(%s) GROUP BY useruuid;""",(country,))
            result = cursor.fetchall()
            total_rows = []
            for row in result:
                user = row[0]
                time = row[1]
                no_rows_for_user = row[2]

                names.append(user)
                time = time.total_seconds()
                average_time = "{0:.2f}".format(time/no_rows_for_user)
                data.append(float(average_time))
                total_rows.append(no_rows_for_user)
            print(sum(total_rows))

        data, names = zip(*sorted(zip(data, names)))
        return data, names

    def get_distinct_feature(self, feature, from_table):
        cursor = self.conn.cursor()
        cursor.execute("""select distinct {} from "{}";""".format(feature, from_table))
        return [feature_name[0] for feature_name in cursor.fetchall()]

    def get_users_in_country(self, country,ratio=0.5):
        cursor = self.conn.cursor()
        cursor.execute("""
            with auxiliary_user_table AS (
                    SELECT COUNT(*) AS Number_in_country, useruuid AS users, country 
                    FROM location 
                    WHERE country!='' 
                    GROUP BY country, useruuid 
                    ORDER BY useruuid), 
            auxiliary_total_sum_for_users AS ( 
                    SELECT SUM(Number_in_country) AS total_sum, users as users1
                    FROM auxiliary_user_table
                    GROUP BY users
                    ORDER BY users),
            auxiliary_user_count_country AS (
                    SELECT Number_in_country, users
                    FROM auxiliary_user_table
                    WHERE country=(%s)
                    ORDER BY users
            ), 

            auxiliary_ratios AS (
                    SELECT auxiliary_user_count_country.Number_in_country AS Number_in_Japan, auxiliary_total_sum_for_users.total_sum AS Total_numbers,
                           (auxiliary_user_count_country.Number_in_country/auxiliary_total_sum_for_users.total_sum) AS Ratio, auxiliary_user_count_country.users AS users3
                    FROM auxiliary_user_count_country INNER JOIN auxiliary_total_sum_for_users ON (auxiliary_user_count_country.users = auxiliary_total_sum_for_users.users1)
                    ORDER BY users3
            )

            SELECT users3 FROM auxiliary_ratios WHERE ratio>=(%s);""",(country,ratio,))
        return [user[0] for user in cursor.fetchall()]


    def get_min_start_time_for_country(self, country):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT MIN(start_time) FROM location WHERE country=(%s);""",(country,))
        return cursor.fetchall()[0][0]

    def get_max_start_time_for_country(self, country):
        cursor = self.conn.cursor()
        cursor.execute("""SELECT MAX(start_time) FROM location WHERE country=(%s);""",(country,))
        return cursor.fetchall()[0][0]



if __name__ == '__main__':
    d = DatabaseHelper()
    print(d.find_cooccurrences("f67ae795-1f2b-423c-ba30-cdd5cbb23662", useruuid2="f3437039-936a-41d6-93a0-d34ab4424a96", asGeoJSON=False))
    #d.dump_missing_geographical_rows()
    #d.drop_tables()
    #d.db_setup()
    #d.insert_all_from_json()
    #d.db_create_indexes()
