#!/usr/bin/env python

from pymongo import GEO2D
from GlobalConfigs import  eateries

eateries.create_index([("eatery_coordinates", GEO2D)]) 
eateries.ensure_index([("eatery_coordinates", pymongo.GEOSPHERE)])

for e in eateries.find({"eatery_coordinates": {"$near": [latitude, longitude]}}).limit(5):
        print e.get("eatery_coordinates"), e.get("eatery_name")



