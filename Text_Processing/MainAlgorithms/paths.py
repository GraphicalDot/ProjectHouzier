#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import sys

path_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_in_memory_classifiers = os.path.join(path_parent_dir + "/PrepareClassifiers/InMemoryClassifiers")
path_trainers_file  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/trainers")
