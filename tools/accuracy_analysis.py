#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt

with open(os.path.join("all_201509.json"),"r") as f:
	september_points = json.loads(f.read())

with open(os.path.join("all_201510.json"),"r") as f:
	october_points = json.loads(f.read())

with open(os.path.join("all_201511.json"),"r") as f:
	november_points = json.loads(f.read())

def create_accuracy_histograms(september_points, october_points, november_points):
	september_points = [point["accuracy"] for point in september_points if point["accuracy"] >= 0]
	october_points = [point["accuracy"] for point in october_points if point["accuracy"] >= 0]
	november_points = [point["accuracy"] for point in november_points if point["accuracy"] >= 0]
	f, axarr = plt.subplots(3, sharex=True)
	axarr[0].hist(september_points, 50, alpha=0.75)
	axarr[0].legend("Accuracy histogram for September 2015")
	axarr[0].set_xlabel("Accuracy")
	axarr[0].set_ylabel("Number of points")
	axarr[0].grid(True)


	axarr[1].hist(october_points, 50, alpha=0.75)
	axarr[1].legend("Accuracy histogram for October 2015")
	axarr[1].set_xlabel("Accuracy")
	axarr[1].set_ylabel("Number of points")
	axarr[1].grid(True)

	axarr[2].hist(november_points, 50, alpha=0.75)
	axarr[2].legend("Accuracy histogram for November 2015")
	axarr[2].set_xlabel("Accuracy")
	axarr[2].set_ylabel("Number of points")
	axarr[2].grid(True)

	plt.show()

def create_accuracy_boxplots(september_points, october_points, november_points):
	september_points = [point["accuracy"] for point in september_points if point["accuracy"] >= 0]
	october_points = [point["accuracy"] for point in october_points if point["accuracy"] >= 0]
	november_points = [point["accuracy"] for point in november_points if point["accuracy"] >= 0]
	months = ["september", "october", "november"]
	plt.subplots_adjust(bottom=0.20)
	plt.boxplot([september_points, october_points, november_points], 0, "bx")
	plt.xticks([1, 2, 3], months, rotation="vertical")
	plt.show()

def analyse(september, october, november):
	september_points = [point["datum"] for point in september if point["datum"] != "WGS84"]
	october_points = [point["datum"] for point in october if point["datum"] != "WGS84"]
	november_points = [point["datum"] for point in november if point["datum"] != "WGS84"]
	print(september_points, october_points, november_points)

analyse(september_points, october_points, november_points)
#create_accuracy_boxplots(september_points, october_points, november_points)