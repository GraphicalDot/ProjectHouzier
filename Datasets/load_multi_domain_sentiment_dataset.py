#Author: kaali
#Date: 16september, 2016
#This process a new data which is available at
#http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

#!/usr/bin/env python

import os
from BeautifulSoup import BeautifulSoup
POLARITY = {"1.0": "poor", "2.0": "poor", "3.0": "neutral", "4.0": "good", "5.0": "good"}


DATA_DIR = "/home/kaali/Downloads/PolarityDatasets/sorted_data_acl/sorted_data_acl/"


def books():
        _path = os.path.join(DATA_DIR, "books")
        _positive_soup = BeautifulSoup(open(os.path.join(_path, "positive.review")))
        _negative_soup = BeautifulSoup(open(os.path.join(_path, "negative.review")))



        for review in _positive_soup.findAll("review"):
                print review.find("review_text").text, POLARITY[review.find("rating").text]
        
        for review in _negative_soup.findAll("review"):
                print review.find("review_text").text, POLARITY[review.find("rating").text]




def dvd():  
        path = os.path.join(DATA_DIR, "dvd")
        _file = open(_path, "rb")
    
    
    
def electronics():  
        path = os.path.join(DATA_DIR, "electronics")
        _file = open(_path, "rb")
    
    
    
    
def kitchen_housewares():
        path = os.path.join(DATA_DIR, "kitchen_&_housewares")
        _file = open(_path, "rb")



if __name__ == "__main__":
        books()
