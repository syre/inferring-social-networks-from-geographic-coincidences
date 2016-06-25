
import sys
import os
import os.path

SPARK_HOME = """spark-1.6.0-bin-hadoop2.6/""" ## PATH TO SPARK

sys.path.append(os.path.join(SPARK_HOME, "python", "lib", "py4j-0.9-src.zip"))
sys.path.append(os.path.join(SPARK_HOME, "python", "lib", "pyspark.zip"))
os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages com.databricks:spark-avro_2.10:2.0.1 pyspark-shell"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"


from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DecimalType, DoubleType, FloatType, ByteType, IntegerType, LongType, ArrayType, StringType

conf = (SparkConf()
         .setMaster("local[*]")
         .setAppName("My app"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)



# load data to dataframe
data = (sqlContext.read.format("com.databricks.spark.avro")
      .load("data/201509/*.avro")
      )



# import python3 style functions
from __future__ import print_function
from __future__ import division



# load data to dataframe
data = (sqlContext.read.format("com.databricks.spark.avro")
      .load("s3://sbdp-source-lifelog/environments/prod/revisions/1/location/yearmonth=201509/*.avro")
      )


# ### Look at data

# I kan se strukturen på data her
data.printSchema()

# kig på data
data.show(10)


# ### Define bin calculation functions and periods


from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DecimalType, DoubleType, FloatType, ByteType, IntegerType, LongType, ArrayType, StringType
from dateutil import parser
import math

grid_boundaries_tuple=(-180, 180, -90, 90)
spatial_resolution_decimals = 3

GRID_MIN_LNG = (grid_boundaries_tuple[0] + 180) * pow(10,spatial_resolution_decimals)
GRID_MAX_LNG = (grid_boundaries_tuple[1] + 180) * pow(10,spatial_resolution_decimals)
GRID_MIN_LAT = (grid_boundaries_tuple[2] + 90) * pow(10,spatial_resolution_decimals)
GRID_MAX_LAT = (grid_boundaries_tuple[3] + 90) * pow(10,spatial_resolution_decimals)

def calculate_spatial_bin(lng, lat):
    lat += 90.0
    lng += 180.0
    lat = math.trunc(lat*pow(10, spatial_resolution_decimals))
    lng = math.trunc(lng*pow(10, spatial_resolution_decimals))
    return (abs(GRID_MAX_LAT - GRID_MIN_LAT) *
            (lat-GRID_MIN_LAT)) + (lng-GRID_MIN_LNG)

def calculate_time_bins(start_time, end_time=None):
    start_time = parser.parse(start_time)
    min_datetime = parser.parse('2015-08-09 00:00:00+02')
    start_bin = int(math.floor(
        ((start_time-min_datetime).total_seconds()/60.0)/60))

    if end_time:
        end_time = parser.parse(end_time)
        end_bin = int(math.ceil(((end_time-min_datetime).total_seconds()/60.0)/60))
    else:
        end_bin = start_bin

    if start_bin == end_bin:
        return [start_bin]
    else:
        return list(range(start_bin, end_bin))

# string reps of periods
first_period_min = "2015-09-01 00:00:00+02:00"
first_period_max = "2015-09-10 23:59:59+02:00"
second_period_min = "2015-09-11 00:00:00+02:00"
second_period_max = "2015-09-20 23:59:59+02:00"
third_period_min = "2015-09-21 00:00:00+02:00"
third_period_max = "2015-09-30 23:59:59+02:00"

# datetime objects of periods
first_period_min_date = parser.parse(first_period_min)
first_period_max_date = parser.parse(first_period_max)
second_period_min_date = parser.parse(second_period_min)
second_period_max_date = parser.parse(second_period_max)
third_period_min_date = parser.parse(third_period_min)
third_period_max_date = parser.parse(third_period_max)

# timebins of periods
first_period_min_bin = calculate_time_bins(first_period_min)[0]
first_period_max_bin = calculate_time_bins(first_period_max)[0]
second_period_min_bin = calculate_time_bins(second_period_min)[0]
second_period_max_bin = calculate_time_bins(second_period_max)[0]
third_period_min_bin = calculate_time_bins(third_period_min)[0]
third_period_max_bin = calculate_time_bins(third_period_max)[0]


# ### Create dataframes: filter by country, accuracy, min and max period
# ### Convert to spatial_bin, time_bins
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import DecimalType, DoubleType, FloatType, ByteType, IntegerType, LongType, ArrayType

#filter by country (Sweden) and start_time and end_time
swe_data = data.filter(data["country"] == 'Sweden').filter(data["start_time"] >= first_period_min_date).filter(data["end_time"] <= third_period_max_date).filter(data["accuracy"] <= 55000).filter(data["accuracy"] >= 0)
udf_spatial_bin = udf(calculate_spatial_bin, IntegerType())
# add new column spatial_bin
binned_swe_data = swe_data.withColumn("spatial_bin", udf_spatial_bin("longitude", "latitude"))
# add new column time_bins
udf_time_bins = udf(calculate_time_bins, ArrayType(IntegerType()))
binned_swe_data = binned_swe_data.withColumn("time_bins", udf_time_bins("start_time", "end_time"))
# get distinct spatial bin, time bin, useruuid rows only
binned_swe_data_with_spatial = binned_swe_data.select(binned_swe_data["spatial_bin"], explode(binned_swe_data["time_bins"]).alias("time_bin"), binned_swe_data["useruuid"]).distinct()
binned_swe_data = binned_swe_data_with_spatial.select(binned_swe_data_with_spatial["time_bin"], binned_swe_data_with_spatial["useruuid"]).distinct()

# reduce to (spatial_bin, time_bin) -> [users] (CONVERSION TO RDD)
bins_to_users = binned_swe_data_with_spatial.rdd.map(lambda r: ((r[0],r[1]),[r[2]])).reduceByKey(lambda a, b: a+b if b[0] not in a else a)
bins_to_users_counts = bins_to_users.map(lambda r: (r[0], len(r[1])))
bins_to_users_counts_broadcast = sc.broadcast(bins_to_users_counts.collect())

# ### Statistics

swe_data.printSchema()
swe_data.count()


# ### Define criteria for users

timebin_percentage = 0.2
period_1_max_timebins = first_period_max_bin - first_period_min_bin
period_2_max_timebins = second_period_max_bin - second_period_min_bin
period_3_max_timebins = third_period_max_bin - third_period_min_bin

print(period_1_max_timebins)
print(period_2_max_timebins)
print(period_3_max_timebins)


# ### Find users for each period and find users which are common to every period (RDD version)

period_1_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < first_period_max_bin).filter(binned_swe_data["time_bin"] >= first_period_min_bin).select(binned_swe_data["useruuid"]).map(lambda r: ((r),1)).reduceByKey(lambda a, b: a+b)
period_1_users = period_1_users_counts.filter(lambda r: r[1] / float(period_1_max_timebins) >= timebin_percentage).map(lambda r: r[0])
period_2_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < second_period_max_bin).filter(binned_swe_data["time_bin"] >= second_period_min_bin).select(binned_swe_data["useruuid"]).map(lambda r: ((r),1)).reduceByKey(lambda a, b: a+b)
period_2_users = period_2_users_counts.filter(lambda r: r[1] / float(period_2_max_timebins) >= timebin_percentage).map(lambda r: r[0])

period_3_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < third_period_max_bin).filter(binned_swe_data["time_bin"] >= third_period_min_bin).select(binned_swe_data["useruuid"]).map(lambda r: ((r),1)).reduceByKey(lambda a, b: a+b)
period_3_users = period_3_users_counts.filter(lambda r: r[1] / float(period_3_max_timebins) >= timebin_percentage).map(lambda r: r[0])

users_in_all = period_1_users.intersection(period_2_users).intersection(period_3_users)
users_in_all = sc.broadcast([x["useruuid"] for x in users_in_all.collect()])


# ### Find users for each period and find users which are common to every period (Dataframe version)
period_1_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < first_period_max_bin).filter(binned_swe_data["time_bin"] >= first_period_min_bin).select(binned_swe_data["useruuid"]).groupBy(binned_swe_data["useruuid"]).count()
period_1_users = period_1_users_counts.filter(period_1_users_counts["count"]/float(period_1_max_timebins) >= timebin_percentage).select(binned_swe_data["useruuid"])

period_2_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < second_period_max_bin).filter(binned_swe_data["time_bin"] >= second_period_min_bin).select(binned_swe_data["useruuid"]).groupBy(binned_swe_data["useruuid"]).count()
period_2_users = period_2_users_counts.filter(period_2_users_counts["count"]/float(period_2_max_timebins) >= timebin_percentage).select(binned_swe_data["useruuid"])

period_3_users_counts = binned_swe_data.filter(binned_swe_data["time_bin"] < third_period_max_bin).filter(binned_swe_data["time_bin"] >= third_period_min_bin).select(binned_swe_data["useruuid"]).groupBy(binned_swe_data["useruuid"]).count()
period_3_users = period_3_users_counts.filter(period_3_users_counts["count"]/float(period_3_max_timebins) >= timebin_percentage).select(binned_swe_data["useruuid"])

users_in_all = period_1_users.intersect(period_2_users).intersect(period_3_users)
users_in_all = sc.broadcast([x["useruuid"] for x in users_in_all.collect()])

period_1_users_counts.orderBy("count", ascending=False).show(5)


# ### Divide into periods


period_1_bins_to_users = bins_to_users.filter(lambda row: row[0][1] >= first_period_min_bin
                                and row[0][1] < first_period_max_bin)
period_2_bins_to_users = bins_to_users.filter(lambda row: row[0][1] >= second_period_min_bin
                                and row[0][1] < second_period_max_bin)
period_3_bins_to_users = bins_to_users.filter(lambda row: row[0][1] >= third_period_min_bin
                                and row[0][1] < third_period_max_bin)


# ### Generate co-occurences

# generate cooccurrences in form of: (user1,user2) -> [(spatial,time)]
from itertools import combinations
def generate_cooccurrences(row):
    return [(tuple(sorted(pair)),[row[0]]) for pair in combinations(row[1], 2) if pair[0] in users_in_all.value and pair[1] in users_in_all.value]
    
coocs_1 = period_1_bins_to_users.flatMap(generate_cooccurrences).reduceByKey(lambda a,b: a+b)
coocs_2 = period_2_bins_to_users.flatMap(generate_cooccurrences).reduceByKey(lambda a,b: a+b)
coocs_3 = period_3_bins_to_users.flatMap(generate_cooccurrences).reduceByKey(lambda a,b: a+b)


# ### Auxiliary functions

from dateutil.relativedelta import relativedelta
from pytz import timezone
import datetime

def calculate_datetime(time_bin):
    min_datetime = parser.parse('2015-08-09 00:00:00+02')
    datetime = min_datetime+relativedelta(seconds=60*60*int(time_bin))
    return datetime


# ### Generate is_weekend
def is_weekend(datetime, timezonestring="Europe/Stockholm"):
    converted = datetime.astimezone(timezone(timezonestring))
    return converted.date().isoweekday in [6, 7] or (converted.date().isoweekday == 5 and converted.hour > 18)

def calculate_number_of_weekend_coocs(cooc_arr):
    return sum([is_weekend(calculate_datetime(cooc[1])) for cooc in cooc_arr])


# ### Generate number of evenings
def is_evening(datetime, timezonestring="Europe/Stockholm"):
    converted = datetime.astimezone(timezone(timezonestring))
    return converted.hour > 18

def calculate_number_of_evening_coocs(cooc_arr):
    return sum([is_evening(calculate_datetime(cooc[1])) for cooc in cooc_arr])


# ### Coocs w

def calculate_coocs_w(cooc_arr):
    if len(cooc_arr) == 0:
        return 0
    coocs_w_values = []
    for row in cooc_arr:
        for bins_users in bins_to_users_counts_broadcast.value:
            if bins_users[0][0]==row[0] and bins_users[0][1]==row[1]:
                coocs_w_value = bins_users[1]
                break
                
        coocs_w_values.append(1/float((coocs_w_value-1)))
    return sum(coocs_w_values)/len(cooc_arr)


# ### Generate location entropies
import numpy as np
# generate location entropies (H_l) for use in weighted frequency in form key:spatial_bin, val: H_l
def calculate_H(row):
    H_val = 0
    for user in set(row[1]):
        P_ul = row[1].count(user)/float(len(row[1]))
        H_val += P_ul*np.log2(P_ul)
    return row[0],-H_val
# COLLECTS
period_1_h_vals = sc.broadcast(period_1_bins_to_users.map(lambda row: (row[0][0],row[1])).reduceByKey(lambda a, b: a+b).map(calculate_H).collectAsMap())
period_2_h_vals = sc.broadcast(period_2_bins_to_users.map(lambda row: (row[0][0],row[1])).reduceByKey(lambda a, b: a+b).map(calculate_H).collectAsMap())


# ### Generate features

from pyspark.mllib.regression import LabeledPoint
import pyspark
import numpy as np
from itertools import permutations

# COLLECTS
y_1_users = sc.broadcast(coocs_2.map(lambda row: row[0]).collect())
y_2_users = sc.broadcast(coocs_3.map(lambda row: row[0]).collect())

if isinstance(period_1_users_counts, pyspark.rdd.PipelinedRDD):
    period_1_users_counts_bc = sc.broadcast(period_1_users_counts.collectAsMap())
    period_2_users_counts_bc = sc.broadcast(period_2_users_counts.collectAsMap())
else:
    period_1_users_counts_bc = sc.broadcast(period_1_users_counts.rdd.collectAsMap())
    period_2_users_counts_bc = sc.broadcast(period_2_users_counts.rdd.collectAsMap())


# generate dict of useruuid : coocs_users 
# unique by position, not value, thus user1 : user2 and user2: user1 can occur
def generate_cooccurrences_with_duplicates(row):
    return [(pair[0],[pair[1]]) for pair in permutations(row[1],2) if pair[0] in users_in_all.value and pair[1] in users_in_all.value]

dup_coocs_1 = period_1_bins_to_users.flatMap(generate_cooccurrences_with_duplicates).reduceByKey(lambda a,b: a+b)
dup_coocs_2 = period_2_bins_to_users.flatMap(generate_cooccurrences_with_duplicates).reduceByKey(lambda a,b: a+b)
dup_coocs_1 = sc.broadcast(dup_coocs_1.collectAsMap())
dup_coocs_2 = sc.broadcast(dup_coocs_2.collectAsMap())

    
def compute_mutual_cooccurrences(row, dup_coocs):
    user1 = row[0][0]
    user2 = row[0][1]
    user1_coocs = set(dup_coocs[user1])
    user2_coocs = set(dup_coocs[user2])
    return len(user1_coocs & user2_coocs)/len(user1_coocs | user2_coocs)

def compute_specificity(row, counts):
    user1_count = counts[row[0][0]]
    user2_count = counts[row[0][1]]
    return len(row[1])/(user1_count+user2_count)

def compute_weighted_frequency(row, h_vals):
    spatial_bins = [r[0] for r in row[1]]
    wf_value = 0
    for sb in set(spatial_bins):
        wf_value += spatial_bins.count(sb)*np.exp(-h_vals.value[sb])
    return wf_value

# row = [(user1, user2), [(spatial, timebin), (...), ...]]
def compute_features(y, row, h_vals, counts, dup_coocs):
    # number of cooccurrences
    num_coocs = len(row[1])
    # number of unique (by spatial bin) cooccurrences
    num_unique_coocs = len(set([r[0] for r in row[1]]))
    spatial_bins = [r[0] for r in row[1]]
    # weighted frequency
    weighted_frequency = compute_weighted_frequency(row, h_vals)
    # diversity
    diversity = np.exp(-np.sum([spatial_bins.count(sb)/float(len(spatial_bins))*np.log2(spatial_bins.count(sb)/float(len(spatial_bins))) for sb in set(spatial_bins)]))
    # sum weekend
    sum_weekends = calculate_number_of_weekend_coocs(row[1])
    # sum evenings
    sum_evenings = calculate_number_of_evening_coocs(row[1])
    # coocs_w
    coocs_w = calculate_coocs_w(row[1])
    specificity = compute_specificity(row, counts)
    mutual_cooccurrences = compute_mutual_cooccurrences(row, dup_coocs)
    return LabeledPoint(y, [num_coocs, num_unique_coocs, diversity,
                            weighted_frequency, sum_weekends, sum_evenings, coocs_w, mutual_cooccurrences, specificity])

def compute_train_features(row):
    y = 1 if row[0] in y_1_users.value else 0
    return compute_features(y, row, period_1_h_vals, period_1_users_counts_bc.value, dup_coocs_1.value)
def compute_test_features(row):
    y = 1 if row[0] in y_2_users.value else 0
    return compute_features(y, row, period_2_h_vals, period_2_users_counts_bc.value, dup_coocs_2.value)

X_train = coocs_1.map(compute_train_features)

X_test = coocs_2.map(compute_test_features)

from pyspark.mllib.util import MLUtils
X_train = MLUtils.loadLibSVMFile(sc, "data/vedran_thesis_students/X_train_nonfilter", False)
X_test = MLUtils.loadLibSVMFile(sc, "data/vedran_thesis_students/X_test_nonfilter", False)
X_train = X_train.coalesce(1)
X_test = X_test.coalesce(1)
MLUtils.saveAsLibSVMFile(X_train, "X_train_nonfilter_merged")
MLUtils.saveAsLibSVMFile(X_test, "X_test_nonfilter_merged")


# ### Show number of meets and dont meets
X_train_neg = X_train.filter(lambda row: row.label == 0).count()
X_train_pos = X_train.filter(lambda row: row.label == 1).count()
X_test_neg = X_test.filter(lambda row: row.label == 0).count()
X_test_pos = X_test.filter(lambda row: row.label == 1).count()
print("X_train has {} negative samples".format(X_train_neg))
print("X_train has {} positive samples".format(X_train_pos))
print("X_test has {} negative samples".format(X_test_neg))
print("X_test has {} positive samples".format(X_test_pos))


# ### Save to disk
from pyspark.mllib.util import MLUtils

MLUtils.saveAsLibSVMFile(X_train, path+"X_train")
MLUtils.saveAsLibSVMFile(X_test, path+"X_test")


# ### Undersampling

if X_train_pos > X_train_neg:
    ratio = X_train_neg/X_train_pos
    X_train = X_train.filter(lambda row: row.label == 0).union(X_train.filter(lambda row: row.label == 1).sample(False, ratio))
else:
    ratio = X_train_pos/X_train_neg
    X_train = X_train.filter(lambda row: row.label == 1).union(X_train.filter(lambda row: row.label == 0).sample(False, ratio))

if X_test_pos > X_test_neg:
    ratio = X_test_neg/X_test_pos
    X_test = X_test.filter(lambda row: row.label == 0).union(X_test.filter(lambda row: row.label == 1).sample(False, ratio))
else:
    ratio = X_test_pos/X_test_neg
    X_test = X_test.filter(lambda row: row.label == 0).union(X_test.filter(lambda row: row.label == 1).sample(False, ratio))

X_train_neg = X_train.filter(lambda row: row.label == 0).count()
X_train_pos = X_train.filter(lambda row: row.label == 1).count()
X_test_neg = X_test.filter(lambda row: row.label == 0).count()
X_test_pos = X_test.filter(lambda row: row.label == 1).count()
print("X_train has {} negative samples".format(X_train_neg))
print("X_train has {} positive samples".format(X_train_pos))
print("X_test has {} negative samples".format(X_test_neg))
print("X_test has {} positive samples".format(X_test_pos))


# ### Train model and get AUC

# In[ ]:

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.util import MLUtils

# Train model and compute AUC
model = RandomForest.trainClassifier(X_test, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=1000, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32, seed=0)


predictions = model.predict(X_train.map(lambda x: x.features))

labels = X_train.map(lambda x: x.label)

predictionAndLabels = predictions.zip(labels)

metrics = BinaryClassificationMetrics(predictionAndLabels)

print("Area under ROC = {}".format(metrics.areaUnderROC))
print("Area under Precision-Recall Curve = {}".format(metrics.areaUnderPR))
metrics = MulticlassMetrics(predictionAndLabels)

for label in sorted(labels.distinct().collect()):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

