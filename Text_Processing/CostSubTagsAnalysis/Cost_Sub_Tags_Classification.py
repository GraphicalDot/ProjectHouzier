#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 6th February, 2015
Purpose:
    Class CostSubTagClassfier
    This is the class which deals with the cost sub tag classifiation like 
        expensive, cheap, vfm

"""
import sys
import os

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(directory))
from MainAlgorithms import InMemoryMainClassifier, timeit, cd, path_parent_dir, path_trainers_file, path_in_memory_classifiers

class CostSubTagClassifier(InMemoryMainClassifier):
        def __init__(self):
                tag_list = ["cheap", "vfm", "expensive", "null"]
                InMemoryMainClassifier.__init__(self, tag_list)

