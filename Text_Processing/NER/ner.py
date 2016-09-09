#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 4th february, 2015
This file has a NERs class which extarcts the name, entity out of the text
"""
import os
import sys
import subprocess
import warnings
import os
import jsonrpclib
from simplejson import loads
import hashlib
import subprocess
import ConfigParser
import imp



file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(file_path)) 

config = ConfigParser.RawConfigParser()



os.chdir(file_path)
config.read("variables.cfg")
SERVER = jsonrpclib.Server("http://%s:%s"%(config.get("corenlpserver", "ip"), config.getint("corenlpserver", "port")))

class NERs:
    
        def __init__(self, list_of_sentences):
                """"
                list_of_sentences: 
                    geberally belongs to the same review, which is the list of sentences generated from 
                    a review after sentence tokenization
                
                Output will be in the form 
                [[u'reminds you of London or New York bars ...', [u'New York', u'London']]]


                """
                self.list_of_sentences = list_of_sentences
                self.result = list()


        def run(self):
                for sent in self.list_of_sentences:
                        result = loads(SERVER.parse(sent))
                        __result =  self.find_location([(e[0], e[1].get('NamedEntityTag')) for e in result["sentences"][0]["words"]])
                        if __result:
                                self.result.append([sent, __result])

                return self.result        
    
	def find_location(self, __list):
                location_list = list()
                i = 0
                for __tuple in __list:
                        if __tuple[1] == "LOCATION":
                                location_list.append([__tuple[0], i])

                        i += 1


                i = 0
                try:
                        new_location_list = list()
                        [first_element, i] = location_list.pop(0)
                        new_location_list.append([first_element])
                        for element in location_list:
                                if i == element[1] -1:
                                        new_location_list[-1].append(element[0])
                
                                else:
                                        new_location_list.append([element[0]])
                        
                                i = element[1]


                        return list(set([" ".join(element) for element in new_location_list]))
                except Exception as e:
                        return None


            






