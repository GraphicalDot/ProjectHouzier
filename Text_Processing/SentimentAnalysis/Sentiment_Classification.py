#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys
import os
import time
import inspect
import itertools
import numpy as np
from sklearn.externals import joblib





#Changing path to the main directory where all the modules are lying
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(directory))
from MainAlgorithms import InMemoryMainClassifier, timeit, cd, path_parent_dir, path_trainers_file, path_in_memory_classifiers




class SentimentClassifier(InMemoryMainClassifier):
	def __init__(self):
		"""
                If the data is to be loaded from the files, Then supply from_files= True while
                initiating InMemoryMainClassifier class, By Default it is false
                """
                tag_list = ["super-positive", "positive", "negative", "super-negative", "neutral"] 
		InMemoryMainClassifier.__init__(self, tag_list)
		


