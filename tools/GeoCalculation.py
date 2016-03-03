#!/usr/bin/env python3

from geopy.distance import vincenty

class GeoCalculation(object):
    """docstring for GeoCalculation"""
    def __init__(self):
        pass


    def distance_between(self, point1, point2):
        return vincenty(point1, point2).meters
