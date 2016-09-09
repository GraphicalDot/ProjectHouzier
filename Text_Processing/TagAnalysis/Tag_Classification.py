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




class TagClassifier(InMemoryMainClassifier):
	def __init__(self):
		tag_list = ["food", "ambience", "cost", "service", "overall", "null"]
		InMemoryMainClassifier.__init__(self, tag_list)  	



