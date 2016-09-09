#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author:Kaali
Dated: 17 January, 2015
Day: Saturday
Description: This file has been written for the android developer, This will be used by minimum viable product implementation
            on android 

Comment: None
"""


from __future__ import absolute_import
import copy
import re
import csv
import codecs
from textblob import TextBlob 
import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.autoreload
from tornado.httpclient import AsyncHTTPClient
from tornado.log import enable_pretty_logging
import hashlib
import subprocess
import shutil
import json
import os
import StringIO
import difflib
from textblob.np_extractors import ConllExtractor 
from bson.json_util import dumps
from Text_Processing import NounPhrases, get_all_algorithms_result, RpRcClassifier, \
		bcolors, CopiedSentenceTokenizer, SentenceTokenizationOnRegexOnInterjections, get_all_algorithms_result, \
		path_parent_dir, path_trainers_file, path_in_memory_classifiers, timeit, cd, SentimentClassifier, \
		TagClassifier, NERs, NpClustering
from compiler.ast import flatten
from topia.termextract import extract
from Text_Processing import WordTokenize, PosTaggers, NounPhrases
import decimal
import time
from datetime import timedelta
import pymongo
from collections import Counter
from functools import wraps
import itertools
import random
from sklearn.externals import joblib
import numpy
from multiprocessing import Pool
import base64
import requests
from PIL import Image
import inspect
import functools
import tornado.httpserver
from itertools import ifilter
from tornado.web import asynchronous
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from bson.son import SON
from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
from GlobalConfigs import connection, eateries, reviews, yelp_eateries, yelp_reviews, eateries_results_collection,\
                    elasticsearch, users_details, users_feedback, users_queries
         

from ProductionEnvironmentApi.text_processing_api import PerReview, EachEatery, DoClusters
from ProductionEnvironmentApi.text_processing_db_scripts import MongoScriptsReviews, MongoScriptsEateries, \
            MongoScriptsDoClusters, MongoScripts
from ProductionEnvironmentApi.prod_heuristic_clustering import ProductionHeuristicClustering
from ProductionEnvironmentApi.join_two_clusters import ProductionJoinClusters
from ProductionEnvironmentApi.elasticsearch_db import ElasticSearchScripts
from ProductionEnvironmentApi.query_resolution import QueryResolution



from ProcessingCeleryTask import MappingListWorker, PerReviewWorker, EachEateryWorker, DoClustersWorker     


def print_execution(func):
        "This decorator dumps out the arguments passed to a function before calling it"
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        fname = func.func_name
        def wrapper(*args,**kwargs):
                start_time = time.time()
                print "{0} Now {1} have started executing {2}".format(bcolors.OKBLUE, func.func_name, bcolors.RESET)
                result = func(*args, **kwargs)
                print "{0} Total time taken by {1} for execution is --<<{2}>>--{3}\n".format(bcolors.OKGREEN, func.func_name,
                                (time.time() - start_time), bcolors.RESET)
                return result
        return wrapper




def cors(f):
        @functools.wraps(f) # to preserve name, docstring, etc.
        def wrapper(self, *args, **kwargs): # **kwargs for compability with functions that use them
                self.set_header("Access-Control-Allow-Origin",  "*")
                self.set_header("Access-Control-Allow-Headers", "content-type, accept")
                self.set_header("Access-Control-Max-Age", 60)
                return f(self, *args, **kwargs)
        return wrapper
                                                        




def time_series(__result):
        n_result = list()

        def __a(__dict, dates):
            __result = []
            for date in dates:
                    if __dict[date]:
                            __result.append(__dict[date])
                    else:
                            __result.append(0)
            return __result

        for element in __result:
                n_result.append([str(element[0].replace("-", "")), str(element[1].split(" ")[0])])
        
        sentiments, dates = zip(*n_result)
        dates = sorted(list(set(dates)))
        
        neutral = __a(Counter([x[1].split(" ")[0] for x in ifilter(lambda x: x[0] == "neutral" , n_result)]), dates)
        superpositive = __a(Counter([x[1].split(" ")[0] for x in ifilter(lambda x: x[0] == "superpositive" , n_result)]), dates)
        supernegative = [-abs(num) for num in __a(Counter([x[1].split(" ")[0] for x in ifilter(lambda x: x[0] == "supernegative" , n_result)]), dates)]
        negative = [-abs(num) for num in __a(Counter([x[1].split(" ")[0] for x in ifilter(lambda x: x[0] == "negative" , n_result)]), dates)]
        positive = __a(Counter([x[1].split(" ")[0] for x in ifilter(lambda x: x[0] == "positive" , n_result)]), dates)

        series = [{"name": e[0], "data": eval(e[0]), "color": e[1]} for e in [("neutral", "#ADB8C2"), ("superpositive", "green"), ("supernegative", "#B46254"), ("positive", "#598C73"), ("negative", "#8B7BA1")]]
        
        cumulative = numpy.cumsum([sum(e) for e in zip(negative, supernegative, superpositive, positive, neutral)])
        return {"categories": dates,
                "series": series, 
                "cumulative": [{"name": "cumulative", "data": list(cumulative), "color": "LightSlateGray"}]}


def convert_for(data):
                        highchart_categories = []
                        supernegative, superpositive, negative, neutral, positive = [], [], [], [], []
                        if type(data) == list:
                                for __data in data:
                            
                                        highchart_categories.append(__data.get("name"))
                                        supernegative.append(__data.get("super-negative"))
                                        superpositive.append(__data.get("super-positive"))
                                        negative.append(__data.get("negative"))
                                        positive.append(__data.get("positive"))
                                        neutral.append(__data.get("neutral"))
                        
                        if type(data) == dict:
                            for name, __data in data.iteritems():
                                        highchart_categories.append(name)
                                        supernegative.append(__data.get("super-negative"))
                                        superpositive.append(__data.get("super-positive"))
                                        negative.append(__data.get("negative"))
                                        positive.append(__data.get("positive"))
                                        neutral.append(__data.get("neutral"))

                        highchart_series = [
                                    {"name": "supernegative", "data": supernegative, 'color': "#B46254"},
                                    {"name": "negative", "data": negative, 'color': "#8B7BA1"},
                                    {"name": "neutral", "data": neutral, 'color': "#ADB8C2"},
                                    {"name": "positive", "data": positive, 'color': "#598C73"},
                                    {"name": "superpositive", "data": superpositive, 'color': "green"}, 
                                    ]

                        return {"categories": highchart_categories, "series": highchart_series}

class UsersFeedback(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                
                feedback = self.get_argument("feedback")
                name = self.get_argument("name")
                telephone = self.get_argument("telephone")
                email = self.get_argument("email")
                print feedback
                users_feedback.insert({"feedback": feedback, "name": name, "telephone": telephone, "email": email, "timestamp": time.time()})
                self.write({"success": True,
			"error": False,
			})
                self.finish()
                return

class UsersDetails(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                fb_id = self.get_argument("id")
                name = self.get_argument("name")
                email = self.get_argument("email")
                picture = self.get_argument("picture")
                print fb_id, name, email, picture
                print users_details
                print users_details.update({"fb_id": fb_id}, {"$set": { "name": name, "email": email, "picture": picture}}, upsert=True)
                self.write({"success": True,
			"error": False,
			})
                self.finish()
                return


class LimitedEateriesList(tornado.web.RequestHandler):
	@cors
	@print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def get(self):
                """
                This gives only the limited eatery list like the top on the basis of the reviews count
                """
                projection={"eatery_id": True, "eatery_name": True, "eatery_address": True, "eatery_coordinates": True, "eatery_total_reviews": True, "_id": False}
                result = [eatery for eatery in list(eateries.find({"eatery_area_or_city": "ncr"},  projection).limit(100).sort("eatery_total_reviews", -1)) if eatery.get("eatery_coordinates")]
                self.write({"success": True,
			"error": False,
                        "result": result,
			})
                self.finish()





class EateriesOnCharacter(tornado.web.RequestHandler):
	@cors
	@print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def post(self):
                """
                Returns eateries on the basis of the character starting the name of the eatery
                
                """
                page_num = int(self.get_argument("page_num"))
                skip = page_num*10
                projection={"eatery_id": True, "eatery_name": True, "eatery_address": True, "eatery_coordinates": True, "_id": False, "trending_factor": True}
                result = [eatery for eatery in list(eateries.find({"eatery_area_or_city": "ncr"},  projection).skip(skip).limit(10).sort("eatery_total_reviews", -1)) \
                        if eatery.get("eatery_coordinates")]
                

                def highest_trending(eatery_data, category):
                        result = sorted([[eatery_data[category][key].get("trending_factor"), key] for key in eatery_data[category].keys()], reverse=True, key=lambda x: x[0])
                        if not "null" in result[0][1].split("-"):
                                return result[0][1]
                        return result[1][1]



                for eatery in result:
                        eatery_data = eateries_results_collection.find_one({"eatery_id": eatery.get("eatery_id")})
                        sorted_by_trending = sorted(eatery_data["food"]["dishes"], reverse=True, key = lambda x: x.get("trending_factor"))
                        
                        if eatery_data:
                                    try:
                                            eatery.update({"trending1": sorted_by_trending[0].get("name")})
                                            eatery.update({"trending2": sorted_by_trending[1].get("name")})
                                    except Exception as e:
                                            print e
                                            eatery.update({"trending1": "Not enough data"})
                                            eatery.update({"trending2": "Not enough data"})
                                            
                                    eatery.update({"cost": highest_trending(eatery_data, "cost")})
                                    eatery.update({"service": highest_trending(eatery_data, "service")})
                                    eatery.update({"ambience": highest_trending(eatery_data, "ambience")})
                        else:
                                eatery.update({"trending1": "abc"})
                                eatery.update({"trending2": "def"})
               
                print result
                self.write({"success": True,
			"error": False,
                        "result": result,
			})
                self.finish()


                


#TODO : Tornadoright now hangs and slows down for another requests, if any one of the request fails
##Return something if a requests cannot be completed before a certain time limit
class GetWordCloud(tornado.web.RequestHandler):
        @property
        def executor(self):
                return self.application.executor
    
    
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                Args:
                    eatery_id
                    category
                """
            
                eatery_id = self.get_argument("eatery_id")
                category = self.get_argument("category")


                        
                if not bool(reviews.find({"eatery_id": eatery_id}).count()):
                        self.set_status(400)
                        self.write({"error": True, "success": False, "error_messege": "The eatery id  {0} is not present".format(eatery_id),})
                        self.finish()
                        return 

                if reviews.find({"eatery_id": eatery_id}).count() == 0:
                        self.set_status(400)
                        self.finish({"error": True, "success": False, "error_messege": "The eatery id  {0} has no reviews prsent in the database".format(eatery_id),})
                        return 

                category = category.lower()
                #name of the eatery
                eatery_name = eateries.find_one({"eatery_id": eatery_id}).get("eatery_name")
                if category not in ["service", "food", "ambience", "cost"]:
                        self.set_status(400)
                        self.finish({"error": True, "success": False, "error_messege": "This is a n invalid tag %s"%category,})
                        return 
       
                """
                if start_epoch and end_epoch:
                        review_list = [(post.get("review_id"), post.get("review_text")) for post in 
                            reviews.find({"eatery_id" :eatery_id, "converted_epoch": {"$gt":  start_epoch, "$lt" : end_epoch}})]
                else:
                        review_list = [(post.get("review_id"), post.get("review_text")) for post in reviews.find({"eatery_id" :eatery_id})] 
                """
                print "Processing word cloud"
                __result = yield self._exe(eatery_id, category)
                
                if category != "food":
                        new_list = list()
                        for key, value in __result.iteritems():
                                value.update({"name": key})
                                new_list.append(value)
                        __result = new_list

                
                
                for element in __result:
                        element.update({"superpositive": element.get("super-positive") })
                        element.update({"supernegative": element.get("super-negative")})
                        element.pop("super-negative")
                        element.pop("super-positive")
               

                self.write({"success": True,
			"error": False,
			"result": __result,
                        })
                self.finish()
                return 


        @run_on_executor
        def _exe(self, eatery_id, category):
                celery_chain = (EachEateryWorker.s(eatery_id)| MappingListWorker.s(eatery_id, PerReviewWorker.s()))()


                while celery_chain.status != "SUCCESS":
                        pass

                try:
                        for __id in celery_chain.children[0]:
                                while __id.status != "SUCCESS":
                                        pass
                except IndexError as e:
                        pass


                do_cluster_result = DoClustersWorker.apply_async(args=[eatery_id])

                while do_cluster_result.status != "SUCCESS":
                        pass
                eatery_instance = MongoScriptsEateries(eatery_id)
                result = eatery_instance.get_noun_phrases(category, 40)
                return result

class UpdateClassifier(tornado.web.RequestHandler):
        @cors
        @timeit
        def post(self):
                """
                Update the classifier with new data into the InMemoryClassifiers folder
                args = update_classifiers.parse_args()    
                whether_allowed = False
                """
                
                if not whether_allowed:
                        return {"success": False,
                                "error": True,
                                "messege": "Right now, Updating Tags or sentiments are not allowed",
                                }


                
                return {"success": True,
                        "error": False,
                        "messege": "Updated!!!",
                        }



class ChangeTagOrSentiment(tornado.web.RequestHandler):
        @cors
        @timeit
        def post(self):
                """
                Updates a sentece with change tag or sentiment from the test.html,
                as the sentences will have no review id, the review_id will be marked as misc and will be stored in 
                training_sentiment_collection or training_tag_collection depending upon the tag or seniment being updated
                """
                args = change_tag_or_sentiment_parser.parse_args()    
                sentence = args["sentence"]
                value = args["value"]
                whether_allowed = True


                print sentence, value
                if not whether_allowed:
                        return {"success": False,
                                "error": True,
                                "messege": "Right now, Updating Tags or sentiments are not allowed",
                                }


                __collection = connection.training_data.training_sentiment_collection
                __collection.insert({"review_id": "misc", "sentence": sentence, "sentiment": value, "epoch_time": time.time(), 
                                "h_r_time": time.asctime()})
                return {"success": True,
                        "error": False,
                        "messege": "Updated!!!",
                        }


class GetTrending(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                """
                        
                latitude = float(self.get_argument("lat"))
                longitude = float(self.get_argument("lng"))
                print type(longitude)
                result = ElasticSearchScripts.get_trending(latitude, longitude)
 
                for __category in ["food", "service", "cost", "ambience"]:
                        for __list in result[__category]:
                                    superpositive = __list.pop("super-positive")
                                    supernegative = __list.pop("super-negative")
                                    __list.update({"superpositive": superpositive, "supernegative": supernegative})


 
                self.write({"success": True,
			        "error": False,
			        "result": result,
			        })
                self.finish()
                return 



def process_object(__object):
            """

            """
            superpositive = __object.pop("super-positive")
            supernegative= __object.pop("super-negative")
            totalsentiments = __object.pop("total_sentiments")
            __object.update({"totalsentiments": totalsentiments, "superpositive": superpositive, "supernegative": supernegative})


            __object.update(time_series(__object["timeline"]))
            __object.update({"subcategory": __object["eatery_name"]})
            return __object

class Query(tornado.web.RequestHandler):
        @property
        def executor(self):
                return self.application.executor
    
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
               result returned :
               {'ambience': ['decor'], 'food': {}, 'cost': [], 'service': [], 
                'sentences': {'food': [(u'i want to have awesome chicken tikka t', u'dishes')], 
                'ambience': [(u'would have nice decor .', u'decor')], 'cost': [], 'service': []}}
                """

                text = self.get_argument("text")
               
                try:
                        l_result = {"food": {}, "ambience": {}, "cost": {}, "service": {}}
                        processed_dishes = list()
                        __result = yield self._exe(text)

                        """
                        __result = {"food": {"dishes": [{"match": [], "suggestions": []}, 
                            {"match": [], "suggestions": []}, 
                        ],
                        }
                        }
                        for element in __result["food"]["dishes"]:
                                if type(element.get("match")) == list:
                                        __match = list()
                                        __suggestions = list()
                                        for __dish in element.get("match"):
                                                __match.append(process_object(__dish))
                                        for __dish in element.get("suggestions"):
                                                __suggestions.append(process_object(__dish))
                                        processed_dishes.append({"name": element.get("name"), "match": __match, "suggestions": __suggestions})
                        l_result["food"]["dishes"] = processed_dishes
                         
                        for main_category, __out in __result.iteritems():
                        __list = list()
                        for item in __out:
                        result.update({main_category: __list})


                        for key, value in result.iteritems():
                        for __value in value:
                        """
                        users_queries.insert({"query": text, "result": __result, "timestamp": time.time()})
                        self.write({"success": True,
			        "error": False,
			        "result": __result,
			        })
                    
                except StandardError as e:
                        print e
                        self.write({"success": False,
			        "error": True,
			        "messege": "Some error occurred while processing your query",
			        })

                self.finish()

        
        @run_on_executor
        def _exe(self, text):
                try:
                        query_resolution_instance = QueryResolution(text)
                        result = query_resolution_instance.run()
                        print "Result from  the query resolution"
                        print result
                        
                        #es_instance  = ElasticSearchScripts()
                        #result = es_instance.elastic_query_processing(result)
                        return result
                except Exception as e:
                        print e
                        raise StandardError("The request cannot be completed, the reason being %s"%e)

class NearestEateries(tornado.web.RequestHandler):
	@cors
	@print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def post(self):
                """
                This gives only the limited eatery list like the top on the basis of the reviews count
                """
                
                lat =  float(self.get_argument("lat"))
                long =  float(self.get_argument("long")) 
                
                print self.get_argument("range")
                range = self.get_argument("range")
                if not range:
                        range = 5
                else:
                        range = int(range)
                


                projection={"eatery_id": True, "eatery_name": True, "eatery_address": True, "eatery_coordinates": True, "eatery_total_reviews": True, "_id": False}
                #result = eateries.find({"eatery_coordinates": {"$near": [lat, long]}}, projection).sort("eatery_total_reviews", -1).limit(10)
                result = eateries.find({"eatery_coordinates" : SON([("$near", { "$geometry" : SON([("type", "Point"), ("coordinates", [lat, long]), \
                        ("$maxDistance", range)])})])}, projection).limit(10)

                __result  = list(result)
                print __result
                self.write({"success": True,
			"error": False,
                        "result": __result,
			})
                self.finish()
                
class EateryDetails(tornado.web.RequestHandler):
	@cors
	@print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def post(self):
                """
                NUmber of dishes to be returne is 14 , and the overfood is to be included also


                keys of each dict in food key of result are
                [u'name', 'series', 'cumulative', u'negative', 'supernegative', u'neutral', u'timeline', 'superpositive', 
                'totalsentiments', u'similar', u'positive', 'categories']

                """
                
                result = {'food': {'series': [{'color': '#B46254', 'data': [4, 1, 10, 1, 1, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'name': 'supernegative'}, {'color': '#8B7BA1', 'data': [16, 31, 42, 6, 10, 5, 12, 2, 8, 2, 1, 2, 2, 3, 1, 0, 2, 0, 0, 0], 'name': 'negative'}, {'color': '#ADB8C2', 'data': [180, 189, 46, 39, 26, 18, 16, 20, 11, 14, 10, 11, 9, 9, 13, 11, 3, 9, 4, 4], 'name': 'neutral'}, {'color': '#598C73', 'data': [172, 140, 128, 38, 24, 29, 32, 23, 14, 12, 22, 10, 16, 11, 8, 10, 11, 7, 9, 7], 'name': 'positive'}, {'color': 'green', 'data': [177, 153, 70, 23, 21, 19, 10, 9, 4, 9, 4, 13, 6, 9, 9, 9, 4, 4, 1, 2], 'name': 'superpositive'}], 'categories': [u'galauti kebabs', u'chicken tikka', u'butter chicken', u'chicken satay', u'chicken tandoori', u'dal makhani', u'rajinder da dhaba', u'chicken', u'shahi paneer', u'mutton korma', u'mughlai parantha', u'chicken burra', u'butter naan', u'malai fish tikka', u'mutton barra', u'dahi kebab', u'veg dishes', u'paneer tikka', u'chicken wings', u'chicken kalmi']}, 'ambience': {'series': [{'color': '#B46254', 'data': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'name': 'supernegative'}, {'color': '#8B7BA1', 'data': [0, 6, 32, 28, 0, 21, 0, 0, 0, 0, 0, 0, 0, 20], 'name': 'negative'}, {'color': '#ADB8C2', 'data': [0, 13, 24, 23, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8], 'name': 'neutral'}, {'color': '#598C73', 'data': [0, 20, 62, 37, 0, 10, 1, 0, 0, 1, 0, 0, 0, 14], 'name': 'positive'}, {'color': 'green', 'data': [0, 6, 9, 12, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6], 'name': 'superpositive'}], 'categories': [u'smoking-zone', u'decor', u'ambience-null', u'ambience-overall', u'romantic', u'crowd', u'view', u'open-area', u'dancefloor', u'music', u'sports-props', u'sports', u'sports-screens', u'in-seating']}, 'cost': {'series': [{'color': '#B46254', 'data': [0, 0, 0, 1, 1], 'name': 'supernegative'}, {'color': '#8B7BA1', 'data': [16, 11, 16, 47, 25], 'name': 'negative'}, {'color': '#ADB8C2', 'data': [4, 1, 6, 10, 4], 'name': 'neutral'}, {'color': '#598C73', 'data': [49, 8, 42, 25, 14], 'name': 'positive'}, {'color': 'green', 'data': [36, 1, 20, 5, 4], 'name': 'superpositive'}], 'categories': [u'value for money', u'cost-null', u'cheap', u'expensive', u'not worth']}, 'eatery_address': u'AB 14, Safdarjung Enclave Market, New Delhi', 'service': {'series': [{'color': '#B46254', 'data': [1, 11, 3, 0, 0, 0, 3], 'name': 'supernegative'}, {'color': '#8B7BA1', 'data': [7, 57, 131, 5, 0, 0, 17], 'name': 'negative'}, {'color': '#ADB8C2', 'data': [1, 5, 23, 0, 1, 0, 2], 'name': 'neutral'}, {'color': '#598C73', 'data': [3, 64, 69, 0, 0, 0, 21], 'name': 'positive'}, {'color': 'green', 'data': [1, 40, 7, 0, 1, 0, 5], 'name': 'superpositive'}], 'categories': [u'management', u'service-overall', u'service-null', u'waiting-hours', u'presentation', u'booking', u'staff']}}

                
                self.write({"success": True,
			"error": False,
                        "result": result})
                self.finish()


class GetDishSuggestions(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def get(self):
                """
                """
                        
                dish_name = self.get_argument("query")
                
                result = ElasticSearchScripts.dish_suggestions(dish_name)
                result = list(set(["{0}".format(element["name"]) for element in result]))
                print result 

                self.write({"success": True,
			        "error": False,
			        "options": result,
			        })
                self.finish()
                return 

class GetDishes(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                """
                dish_name = self.get_argument("dish_name")
                
                result = ElasticSearchScripts.get_dishes(dish_name)
                print result
                for __list in result:
                                    superpositive = __list.pop("super-positive")
                                    supernegative = __list.pop("super-negative")
                                    __list.update({"superpositive": superpositive, "supernegative": supernegative})
                self.write({"success": True,
			        "error": False,
			        "result": result,
			        })
                self.finish()
                return 

class GetEatery(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                """
                        
                number_of_dishes = 20
                eatery_name =  self.get_argument("eatery_name")
                result = eateries_results_collection.find_one({"eatery_name": eatery_name})
                if not result:
                        """
                        If the eatery name couldnt found in the mongodb for the popular matches
                        Then we are going to check for demarau levenshetin algorithm for string similarity
                        """



                    
                    
                    
                        return 
                
                dishes = sorted(result["food"]["dishes"], key=lambda x: x.get("total_sentiments"), reverse=True)[0: number_of_dishes]
                overall_food = result["food"]["overall-food"]
                ambience = result["ambience"]
                cost = result["cost"]
                service = result["service"]


                result = {"food": convert_for(dishes),
                                    "ambience": convert_for(ambience), 
                                    "cost": convert_for(cost), 
                                    "service": convert_for(service), 
                                    "eatery_address": result["eatery_address"],
                                    }

                print result
                self.write({"success": True,
			"error": False,
                        "result": result})
                self.finish()

                return 

class GetEaterySuggestions(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def get(self):
                """
                """
                        
                dish_name = self.get_argument("query")
                
                result = ElasticSearchScripts.eatery_suggestions(dish_name)
                result = list(set(["{0}".format(element["eatery_name"]) for element in result]))
                print result
                self.write({"success": True,
			        "error": False,
			        "options": result,
			        })
                self.finish()
                return 


def main():
        http_server = tornado.httpserver.HTTPServer(Application())
        tornado.autoreload.start()
        http_server.listen("8000")
        enable_pretty_logging()
        tornado.ioloop.IOLoop.current().start()


class Application(tornado.web.Application):
        def __init__(self):
                handlers = [
                    (r"/limited_eateries_list", LimitedEateriesList),
                    (r"/get_word_cloud", GetWordCloud),
                    (r"/resolve_query", Query),
                    (r"/get_trending", GetTrending),
                    (r"/nearest_eateries", NearestEateries),
                    (r"/eateries_on_character", EateriesOnCharacter),
                    (r"/users_details", UsersDetails),
                    (r"/users_feedback", UsersFeedback),
                    (r"/get_dishes", GetDishes),
                    (r"/get_eatery", GetEatery),
                    (r"/get_dish_suggestions", GetDishSuggestions),
                    (r"/get_eatery_suggestions", GetEaterySuggestions),
                    (r"/eatery_details", EateryDetails),]
                settings = dict(cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",)
                tornado.web.Application.__init__(self, handlers, **settings)
                self.executor = ThreadPoolExecutor(max_workers=60)



if __name__ == '__main__':
    print "server reloaded Dude"
    main()
