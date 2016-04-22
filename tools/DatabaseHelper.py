#!/usr/bin/env python3
import psycopg2
import json
import math
import os
from collections import defaultdict
from pytz import timezone
import random
import datetime
from dateutil import parser
from tqdm import tqdm
import numpy as np
from FileLoader import FileLoader


class DatabaseHelper():

    """docstring for DatabaseHelper"""

    def __init__(self, path_to_settings="",
                 grid_boundaries_tuple=(-180, 180, -90, 90),
                 spatial_resolution_decimals=3,
                 from_date=datetime.datetime.strptime(
                     "2015-09-01", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo")),
                 to_date=datetime.datetime.strptime(
                     "2015-11-30", "%Y-%m-%d").replace(tzinfo=timezone("Asia/Tokyo"))):

        self.settings_dict = self.load_login(file_name="settings.cfg",
                                             key_split="=",
                                             path=path_to_settings)
        self.conn = psycopg2.connect("host='{}' dbname='{}' user='{}' password= \
            '{}'".format(self.settings_dict["HOSTNAME"],
                         self.settings_dict["DBNAME"],
                         self.settings_dict["USER"],
                         self.settings_dict["PASS"]))
        self.file_loader = FileLoader()
        self.filter_places_dict = {"Sweden": [[(13.2262862, 55.718211), 1000],
                                              [(17.9529121, 59.4050982),1000]],
                                   "Japan": [[(139.743862, 35.630338), 1000]]}
        self.min_datetime = from_date
        self.max_datetime = to_date
        self.spatial_resolution_decimals = spatial_resolution_decimals
        self.GRID_MIN_LNG = (
            grid_boundaries_tuple[0] + 180) * pow(10,
                                                  spatial_resolution_decimals)
        self.GRID_MAX_LNG = (
            grid_boundaries_tuple[1] + 180) * pow(10,
                                                  spatial_resolution_decimals)
        self.GRID_MIN_LAT = (
            grid_boundaries_tuple[2] + 90) * pow(10,
                                                 spatial_resolution_decimals)
        self.GRID_MAX_LAT = (
            grid_boundaries_tuple[3] + 90) * pow(10,
                                                 spatial_resolution_decimals)

        self.CREATE_TABLE_LOCATION = """
            CREATE TABLE "location" (id serial PRIMARY KEY,
                useruuid text NOT NULL,
                start_time timestamptz NOT NULL,
                end_time timestamptz NOT NULL,
                location GEOGRAPHY(POINT,4326),
                altitude INTEGER NOT NULL,
                accuracy INTEGER NOT NULL,
                region text,
                country text,
                area text,
                place text,
                time_bins integer[],
                spatial_bin bigint)"""

        # if database is setup
        if self.db_setup_test():
            all_users = self.get_distinct_feature("useruuid", "location")
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

        Takes a list as input. Generate a random color
        while the color is in that list.
        Return the unique color

        Keyword Arguments:
            allready_gen {list} -- List of colors which (default: {[]})

        Returns:
            [string] -- string representation of the hex color

        Raises:
            NameError -- Raise an exception is the input list is filled,
            hence there is no more free colors to generate
        """

        if len(allready_gen) < (255*255*255):
            r = lambda: random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            while color in allready_gen:
                r = lambda: random.randint(0, 255)
                color = '#%02X%02X%02X' % (r(), r(), r())
            return color
        raise NameError('No more colors left to choose')

    def load_login(self, file_name="login.txt", key_split="##",
                   value_split=",", has_header=False, path=""):
        d = {}
        with open(os.path.join(os.path.dirname(__file__), path, file_name),
                  'r') as f:
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
            if len(d[key]) == 1:
                d[key] = d[key][0]
        return d

    def db_setup_test(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "select exists(select * from information_schema.tables where \
             table_name=%s)", ('location',))
        return cursor.fetchone()[0]

    def db_setup(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS POSTGIS;")
        cursor.execute(self.CREATE_TABLE_LOCATION)
        self.conn.commit()

    def db_create_indexes(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX ON location (start_time)")
        cursor.execute("CREATE INDEX ON location (end_time)")
        cursor.execute("CREATE INDEX ON location using gist (location)")
        cursor.execute("CREATE INDEX ON location (useruuid)")
        cursor.execute("CREATE INDEX ON location (spatial_bin)")
        cursor.execute("CREATE INDEX ON location (time_bins)")
        cursor.execute("CREATE INDEX ON location (country)")
        self.conn.commit()

    def db_teardown(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE location")

    def insert_location(self, row):
        cursor = self.conn.cursor()
        spatial_bin = self.calculate_spatial_bin(
            row["longitude"], row["latitude"])
        time_bins = self.calculate_time_bins(
            row["start_time"], row["end_time"])

        cursor.execute(""" INSERT INTO location (useruuid, start_time,
            end_time, location, altitude, accuracy, region, country, area,
            place, time_bins, spatial_bin)
            VALUES (%s,%s,%s,ST_SetSRID(ST_MakePoint(%s, %s),4326),
            %s,%s,%s,%s,%s,%s,%s,%s)""", (row["useruuid"], row["start_time"],
                                          row["end_time"], row[
                                              "longitude"], row["latitude"],
                                          row["altitude"], row[
                                              "accuracy"], row["region"],
                                          row["country"], row["area"], row[
                                              "name"], time_bins,
                                          spatial_bin))
        self.conn.commit()

    def calculate_spatial_bin(self, lng, lat):
        lat += 90.0
        lng += 180.0
        lat = math.trunc(lat*pow(10, self.spatial_resolution_decimals))
        lng = math.trunc(lng*pow(10, self.spatial_resolution_decimals))
        return (abs(self.GRID_MAX_LAT - self.GRID_MIN_LAT) *
                (lat-self.GRID_MIN_LAT)) + (lng-self.GRID_MIN_LNG)

    def calculate_time_bins(self, start_time, end_time):
        start_time = parser.parse(start_time)
        end_time = parser.parse(end_time)
        min_datetime = parser.parse('2015-08-09 00:00:00+02')
        start_bin = math.floor(
            ((start_time-min_datetime).total_seconds()/60.0)/60)
        end_bin = math.ceil(((end_time-min_datetime).total_seconds()/60.0)/60)
        if start_bin == end_bin:
            return [start_bin]
        else:
            return list(range(start_bin, end_bin))


    def get_distributions_numbers(self, feature, num_bins=20, max_value=0):
        cursor = self.conn.cursor()
        if not max_value:
            cursor.execute("""SELECT max({}) FROM location""".format(feature))
            max_value = int(cursor.fetchone()[0])

        step_size = int(max_value/num_bins)
        end_value = max_value-int(max_value/num_bins)

        query = "SELECT "+", ".join(["count(CASE WHEN {2} >= {0} AND {2} < {1} \
            THEN 1 END)".format(element, element+step_size, feature)
            for element in range(0, end_value, step_size)]) + \
            ", count(CASE WHEN {0} > {1} THEN 1 END)".format(
                feature, max_value-step_size)+" from location"
        cursor.execute(query)
        results = list(cursor.fetchall()[0])

        bucketized = [str(element)+"-"+str(element+step_size)
                      for element in range(0, end_value, step_size)]
        bucketized.extend([str(max_value-step_size)+"<"])

        return {"results": [{"Number": x[0], "Count":x[1], "Order":index} for
                            index, x in enumerate(zip(bucketized, results))],
                'x_axis': "Number", 'y_axis': "Count"}

    def get_distributions_categories(self, feature, num_bins=20):
        if feature == "country":
            cursor = self.conn.cursor()
            cursor.execute(
                """select country, count(*) from location group by country
                order by count(*) desc;""")
            result = cursor.fetchall()
            countries = [row[0] for row in result]
            count = [row[1] for row in result]
            # return with order value =(-count) to sort by count descending
            return {"results": [{"Countries": x[0], "Count": x[1],
                                 "Order":-x[1]}
                                for x in zip(countries, count)],
                    'x_axis': "Countries", 'y_axis': "Count"}

    def auxiliary_function_velocity(self, lst_start_time, lst_end_time,
                                    lst_points):
        previous_point = lst_points[0]
        previous_start_time = lst_start_time[0]
        previous_end_time = lst_end_time[0]

        count = 0
        total_km_hour = 0.0
        for index, end_time in enumerate(lst_end_time[1:], start=1):
            duration = end_time - previous_start_time
            duration = duration.total_seconds()
            if duration > 0.0:
                distance = self.geo_calc.distance_between(
                    lst_points[index], previous_point)
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

    def get_locations_for_numpy(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT useruuid, spatial_bin, time_bins, country
            FROM location;""")
        return cursor.fetchall()

    def get_velocity_for_users(self, country):
        raw_data = defaultdict(dict)
        cursor = self.conn.cursor()
        cursor.execute(
            """select useruuid,  start_time, end_time,
            ST_X(location::geometry), ST_Y(location::geometry) from location
            where country=(%s) order by start_time;""", (country,))
        result = cursor.fetchall()

        for row in result:
            if row[0] not in raw_data:
                raw_data[row[0]]['start_time'] = [row[1]]
                raw_data[row[0]]['end_time'] = [row[2]]
                raw_data[row[0]]['lat_long'] = [(row[4], row[3])]
            else:
                raw_data[row[0]]['start_time'].append(row[1])
                raw_data[row[0]]['end_time'].append(row[2])
                raw_data[row[0]]['lat_long'].append((row[4], row[3]))

        data = []
        names = []
        for user in raw_data:
            data.append(self.auxiliary_function_velocity(raw_data[user][
                        'start_time'], raw_data[user]['end_time'],
                raw_data[user]['lat_long']))
            names.append(user)

        data, names = zip(*sorted(zip(data, names)))
        return data, names

    def get_locations_for_user(self, useruuid, country=None, spatial_bin=None):
        country_query = ""
        spatial_query = ""
        if country:
            country_query = " AND location.country = '{}'".format(country)
        if spatial_bin:
            spatial_query = " AND location.spatial_bin = '{}'".format(
                spatial_bin)
        cursor = self.conn.cursor()
        cursor.execute("""SELECT useruuid, start_time, end_time,
            ST_X(location::geometry), ST_Y(location::geometry) from location
            where location.useruuid = (%s) """ +
                       country_query + spatial_query + """;""", (useruuid,))
        if cursor.rowcount == 0:
            print("no locations for useruuid")
            return
        else:
            locations = cursor.fetchall()
        return locations

    def find_cooccurrences(self, useruuid, points_w_distances=[],
                           useruuid2=None, asGeoJSON=True, min_timebin=None,
                           max_timebin=None):
        """ find all cooccurrences for a user

        find all cooccurrences for a user within a cell_size and time window
        (time_threshold_in_minutes)
        Can also filter cooccurences by geometry point (see
        points_w_distances).
        Here it finds only cooccurences which is outside of these points
        based on the given distance

        Arguments:
            useruuid {string} -- ID of user
            cell_size {float} -- Size of the cell
            time_threshold_in_minutes {integer} -- [description]

        Keyword Arguments:
            points_w_distances {list of lists} -- list of lists with point
                (longitude,latitude) (as tuple) and distance in meter that are
                excluded from cooccurrences (default: {[]})
                Example [[(139.743862,35.630338), 1000]]
            useruuid2 {string} -- optional argument for when you only want
                cooccurrences with that other user
            asGeoJSON {bool} -- determines whether the location should be
                returned as GeoJSON or ST_X and ST_Y

        Returns:
            list -- list cooccurrences with useruuid, start_time, end_time,
                    geojsoned location
        """
        cursor = self.conn.cursor()

        start = query = ""

        if points_w_distances:
            start = "AND NOT ST_DWithin(location, ST_MakePoint("
            query = " AND NOT ST_DWithin(location, ST_MakePoint(".join(
                ["{0}, {1}), {2})". format(element[0][0], element[0][1],
                                           element[1]) for element in
                 points_w_distances])

        second_user_query = ""
        if useruuid2:
            second_user_query = " AND location.useruuid = '{}' ".format(
                useruuid2)

        range_query = ""
        if (min_timebin is not None and max_timebin is not None):
            range_query = " AND {} =< min(location.time_bins) AND {} >= \
            max(location.time_bins)".format(
                min_timebin, max_timebin)
        cursor.execute("""WITH user1_table 
     AS (SELECT location.id,
                useruuid,
                time_bins,
                spatial_bin
        FROM    location
        WHERE   location.useruuid = %s""" + range_query + """)
        SELECT  location.useruuid,
                location.spatial_bin,
                location.time_bins,
                user1_table.time_bins,
                user1_table.useruuid,
                ST_AsGeoJSON(location.location)
        FROM   location
               inner join user1_table
                       ON location.useruuid != user1_table.useruuid
        WHERE  user1_table.time_bins && location.time_bins
               AND user1_table.spatial_bin = location.spatial_bin
       """ + second_user_query + (start + query) + range_query + ";",
                       (useruuid,))

        return cursor.fetchall()

    def find_cooccurrences_within_area(self, spatial_bin, time_bin=None):
        cursor = self.conn.cursor()
        time_bin_part = ""
        if time_bin:
            time_bin_part = "AND {} = ANY(location.time_bins)".format(time_bin)
        cursor.execute("""
                SELECT DISTINCT(useruuid)
                FROM location
                WHERE location.spatial_bin=(%s)""" + time_bin_part + """
                ;""", (spatial_bin,))

        return [row[0] for row in cursor.fetchall()]

    def find_number_of_records_for_location(self, spatial_bin, useruuid=None):
        user_part = ""
        if useruuid:
            user_part = "AND location.useruuid = (%s)"
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM location
            WHERE location.spatial_bin=(%s)""" + user_part + """;""",
                       (spatial_bin,))
        return cursor.fetchall()

    def has_met_in_timebin(self, user1, user2, time_bin):
        cursor = self.conn.cursor()
        cursor.execute("""
                WITH user1_table
                     AS (SELECT location.id,
                                useruuid,
                                time_bins,
                                spatial_bin
                         FROM   location
                         WHERE  location.useruuid =(%s) AND
                            time_bins && ARRAY[(%s)])
                SELECT  location.useruuid,
                        location.spatial_bin,
                        user1_table.spatial_bin AS user1_spatial,
                        location.time_bins,
                        user1_table.time_bins
                FROM   location
                       inner join user1_table
                               ON user1_table.time_bins && location.time_bins
                WHERE  location.useruuid=(%s) AND user1_table.spatial_bin =
                location.spatial_bin""", user1, time_bin, user2)
        if cursor.rowcount >= 1:
            return True
        return False

    def get_distribution_cooccurrences(self, x_useruuid, y_useruuid,
                                       time_threshold_in_minutes=60*24,
                                       cell_size=0.001):

        cooccurrences = self.find_cooccurrences(
            x_useruuid, points_w_distances=[], useruuid2=y_useruuid)

        time_dict = {}
        start = datetime.datetime.strptime("01-09-2015", "%d-%m-%Y")
        end = datetime.datetime.strptime("01-12-2015", "%d-%m-%Y")
        # generate range of dates from start to end with interval of 1 day
        date_generated = [
            start + datetime.timedelta(days=x) for x in range(0, (end-start).
                                                              days)]

        for date in date_generated:
            time_dict[date.strftime("%d/%m/%Y")] = 0
        for cocc in cooccurrences:
            start_date = cocc[1].strftime("%d/%m/%Y")
            end_date = cocc[2].strftime("%d/%m/%Y")
            time_dict[start_date] += 1

        return [{"Date": date_string, "Cooccurrences": value} for date_string,
                value in time_dict.items()]

    def dump_missing_geographical_rows(self):
        cursor = self.conn.cursor()
        query = "select id, ST_X(location::geometry), ST_Y(location::geometry)\
                from location where coalesce(location.country, '') = '';"
        cursor.execute(query)
        records = cursor.fetchall()
        list_of_records = []
        for record in records:
            list_of_records.append(record)

        with open('missing_records.json', 'w') as outfile:
            json.dump(list_of_records, outfile)

    def get_all_cooccurrences_as_network(self, time_threshold_in_minutes=60*24,
                                         cell_size=0.001):
        cursor = self.conn.cursor()

        distinct_users = self.get_distinct_feature("useruuid", "user")
        nodes = []
        edges = []
        count = 0
        for user in distinct_users:
            locations = self.get_locations_for_user(user)
            if not any(node['id'] == user for node in nodes):
                nodes.append(
                    {"id": user, "label": user,
                        "color": self.user_colors[user]})
            cooccurrences = []
            for location in locations:
                start_time = location[1]
                end_time = location[2]
                longitude = location[3]
                latitude = location[4]

            # find coocurrences by taking time_treshold_in_hours/2 before
            #   start_time and time_threshold_in_hours/2 after end_time
            # this also means time window can get really long, what are the
            # consequences?
            cursor.execute(""" SELECT useruuid AS geom from location
                where location.useruuid != (%s)
                and (start_time between (%s) - interval '%s minutes'
                and (%s)) and (end_time between (%s) and (%s) +
                interval '%s minutes') and abs(ST_X(location::geometry)-(%s))
                <= (%s) and abs(ST_Y(location::geometry)-(%s)) <= (%s)""",
                           (user, start_time, time_threshold_in_minutes/2,
                            start_time, end_time, end_time,
                            time_threshold_in_minutes/2, longitude, cell_size,
                            latitude, cell_size))

            result = cursor.fetchall()
            if result:
                cooccurrences.extend(result)
            for cooccurrence in cooccurrences:
                if not any(node['id'] == cooccurrence[0] for node in nodes):
                    nodes.append({"id": cooccurrence[0], "label": cooccurrence[
                                 0],
                        "color": self.user_colors[cooccurrence[0]]})
            edges.extend([{"source": user, "target": element[0], "id":index}
                          for index, element in enumerate(cooccurrences,
                                                          start=count)])
            count += len(cooccurrences)

        return {"nodes": nodes, "edges": edges}

    def get_users_with_most_updates(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "select useruuid from location group by useruuid \
            order by count(*) desc;")
        return [element[0] for element in cursor.fetchall()]

    def get_locations_by_country(self, country, start_datetime, end_datetime):
        cursor = self.conn.cursor()
        cursor.execute(""" SELECT useruuid, ST_AsGeoJSON(location) AS geom,
            start_time, end_time FROM location
            WHERE country=(%s) AND
            ((start_time, end_time) OVERLAPS ((%s), (%s)));""",
                       (country, start_datetime, end_datetime))
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
            cursor.execute(
                """SELECT country, SUM((end_time-start_time)) AS
                total_diff_time, count(*) AS number_rows_for_user
                FROM location GROUP BY country ORDER BY country;""")
            result = cursor.fetchall()
            for row in result:
                country = row[0]
                names.append(country)
                if country != "" and country != " " and country is not None:
                    time = row[1]
                    number_rows_for_country = row[2]

                    total_rows.append(number_rows_for_country)
                    time = time.total_seconds()
                    average_time = "{0:.2f}".format(
                        time/number_rows_for_country)
                    data.append(float(average_time))
                else:
                    print("Tom streng!")
            print("total_rows:")
            print(total_rows)
        else:
            cursor = self.conn.cursor()
            cursor.execute(
                """ SELECT useruuid, SUM((end_time-start_time)) AS
                total_diff_time, count(*) AS number_rows_for_user
                FROM location WHERE country=(%s) GROUP BY useruuid;""",
                (country,))
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
        cursor.execute(
            """select distinct {} from "{}";""".format(feature, from_table))
        return [feature_name[0] for feature_name in cursor.fetchall()]

    def get_users_in_country(self, country, ratio=0.5):
        cursor = self.conn.cursor()
        cursor.execute("""
            with auxiliary_user_table AS (
                    SELECT COUNT(*) AS Number_in_country, useruuid AS users,
                    country
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
                    SELECT auxiliary_user_count_country.Number_in_country AS
                        Number_in_Japan,
                        auxiliary_total_sum_for_users.total_sum AS
                        Total_numbers,
                        (auxiliary_user_count_country.Number_in_country/
                        auxiliary_total_sum_for_users.total_sum) AS Ratio,
                        auxiliary_user_count_country.users AS users3
                    FROM auxiliary_user_count_country INNER JOIN
                    auxiliary_total_sum_for_users ON
                    (auxiliary_user_count_country.users =
                    auxiliary_total_sum_for_users.users1)
                    ORDER BY users3
            )

            SELECT users3 FROM auxiliary_ratios WHERE ratio>=(%s);""",
                       (country, ratio,))
        return [user[0] for user in cursor.fetchall()]

    def get_min_start_time_for_country(self, country):
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT MIN(start_time) FROM location WHERE country=(%s);""",
            (country,))
        return cursor.fetchall()[0][0]

    def get_max_start_time_for_country(self, country):
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT MAX(start_time) FROM location WHERE country=(%s);""",
            (country,))
        return cursor.fetchall()[0][0]

    def generate_numpy_matrix_from_database(self):
        useruuid_dict = {}
        country_dict = {}

        user_count = 0
        country_count = 0
        rows = self.get_locations_for_numpy()
        locations = []

        for row in tqdm(rows):
            useruuid = row[0]
            spatial_bin = row[1]
            time_bins = row[2]
            country = row[3]
            if useruuid not in useruuid_dict:
                user_count += 1
                useruuid_dict[useruuid] = user_count
            if country not in country_dict:
                country_count += 1
                country_dict[country] = country_count
            useruuid = useruuid_dict[useruuid]
            country = country_dict[country]
            for time_bin in time_bins:
                locations.append(
                    [useruuid, spatial_bin, time_bin, country])
        locations = np.array(locations)

        return useruuid_dict, country_dict, locations

    def update_missing_records(self):
        data = self.file_loader.load_missing_data()
        REAL_COUNTRIES = {'Republic of China': 'China',
                          'Islamic Republic of Iran': 'Iran',
                          'Republic of the Philippines': 'Philippines',
                          'RSA': 'South Africa',
                          'Russian Federation': 'Russia',
                          'Spain (territorial waters)': 'Spain',
                          'Territorial waters of Bornholm': 'Denmark',
                          'Territorial waters of Gotland': 'Sweden',
                          'The Netherlands': 'Netherlands',
                          'United States of America': 'United States',
                          'Luxemburg': 'Luxembourg'}
        s = set()
        for record in data:
            try:
                country = record['country']
                spatial_bin = self.calculate_spatial_bin(record['lng'],
                                                         record['lat'])
                if country in REAL_COUNTRIES:
                    country = REAL_COUNTRIES[country]
                s.add((country, spatial_bin))
            except KeyError:
                if 'motel' in record:
                    spatial_bin = self.calculate_spatial_bin(
                        record['lng'], record['lat'])
                    s.add(('South Korea', spatial_bin))
        lst = list(s)
        cursor = self.conn.cursor()
        for element in tqdm(lst):
            cursor.execute("""UPDATE location SET country=%s WHERE country='' AND
             spatial_bin=%s""",
                           (element[0], element[1]))
            self.conn.commit()

    def delete_test_users(self):
        users = ['02cbb276-93e7-4baa-81aa-ea3f5bf6230a',
                 '3eee0fe1-ef56-42a9-9b16-ee0677c079ee',
                 '578669a8-4b85-49a2-bf46-5437d6192252',
                 '7cb9fddd-d3b6-49f8-9259-51b948c2ac1f',
                 '8298d998-684a-4e90-9a1e-cd0164dabf2e',
                 'e8d40f3a-1e07-4c26-adfb-39a8366d4bbd',
                 'ebb04181-3cc2-4fac-a34f-1962a6953081']
        cursor = self.conn.cursor()
        for user in users:
            cursor.execute(""" DELETE FROM location WHERE useruuid=%s""",
                           (user, ))
            self.conn.commit()

if __name__ == '__main__':
    d = DatabaseHelper()
    # d.update_missing_records()
    d.generate_numpy_matrix_from_database()

