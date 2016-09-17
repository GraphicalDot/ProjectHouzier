#Author: kaali
#Date: 16september, 2016
#This process a new data which is available at
#http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

#!/usr/bin/env python

import os
import sys
from BeautifulSoup import BeautifulSoup
POLARITY = {"1.0": "poor", "2.0": "poor", "3.0": "neutral", "4.0": "good", "5.0": "good"}
__dir__ = os.path.dirname(os.path.abspath(__file__))
print  __dir__


#DATA_DIR = "/home/kaali/Downloads/PolarityDatasets/sorted_data_acl/sorted_data_acl/"
print os.path.join(__dir__, "sorted_data_acl")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "sorted_data_acl")
print DATA_DIR

REVIEWS = list()



def fetch_data(path):
        _positive_file = open(os.path.join(path, "positive.review"))
        _negative_file = open(os.path.join(path, "negative.review"))
        _positive_soup = BeautifulSoup(_positive_file)
        _negative_soup = BeautifulSoup(_negative_file)
        for review in _positive_soup.findAll("review"):
                REVIEWS.append([review.find("review_text").text.replace("\n",""),
                                POLARITY[review.find("rating").text]])
        
        for review in _negative_soup.findAll("review"):
                REVIEWS.append([review.find("review_text").text.replace("\n",""),
                                POLARITY[review.find("rating").text]])
        _positive_file.close()
        _negative_file.close()
        return 


def books():
        path = os.path.join(DATA_DIR, "books")
        fetch_data(path)


def dvd():  
        path = os.path.join(DATA_DIR, "dvd")
        fetch_data(path)
    
    
    
def electronics():  
        path = os.path.join(DATA_DIR, "electronics")
        fetch_data(path)
    
    
    
    
def kitchen_housewares():
        path = os.path.join(DATA_DIR, "kitchen_&_housewares")
        fetch_data(path)






if __name__ == "__main__":
        books()
        dvd()
        electronics()
        kitchen_housewares()
        print REVIEWS




