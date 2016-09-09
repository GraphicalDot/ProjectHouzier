#!/usr/bin/env python
#-*- coding: utf-8 -*-


from __future__ import absolute_import
import base64
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
from datetime import datetime, timedelta
from pytz import timezone
import pytz

from compiler.ast import flatten
from topia.termextract import extract
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
from termcolor import cprint 
from pyfiglet import figlet_format
from Crypto.PublicKey import RSA
import jwt
from jwt import _JWTError
import ConfigParser
from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
from connections import eateries, reviews, eateries_results_collection, reviews_results_collection, short_eatery_result_collection, \
        bcolors, users_reviews_collection, users_feedback_collection, users_details_collection, server_address, pictures_collection,\
        users_queried_addresses_collection, users_dish_collection, pictures_collection
        
from ProductionEnvironmentApi.text_processing_api import PerReview, EachEatery, DoClusters
from ProductionEnvironmentApi.text_processing_db_scripts import MongoScriptsReviews, MongoScriptsDoClusters
from ProductionEnvironmentApi.prod_heuristic_clustering import ProductionHeuristicClustering
from ProductionEnvironmentApi.join_two_clusters import ProductionJoinClusters
from ProductionEnvironmentApi.elasticsearch_db import ElasticSearchScripts




print server_address
file_path = os.path.dirname(os.path.abspath(__file__))
parent_dirname = os.path.dirname(os.path.dirname(file_path))

if not os.path.exists("%s/private.pem"%parent_dirname):
        os.chdir(parent_dirname)
        subprocess.call(["openssl", "genrsa", "-out", "private.pem", "1024"])
        subprocess.call(["openssl", "rsa", "-in", "private.pem", "-out", "public.pem", "-outform", "PEM", "-pubout"])
        os.chdir(file_path)

private = open("%s/private.pem"%parent_dirname).read()
public = open("%s/public.pem"%parent_dirname).read()
private_key = RSA.importKey(private)
public_key = RSA.importKey(public)

def print_execution(func):
        "This decorator dumps out the arguments passed to a function before calling it"
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        fname = func.func_name
        def wrapper(*args,**kwargs):
                start_time = time.time()
                print "{0} Now {1} have started executing {2}".format(bcolors.OKBLUE, func.func_name, bcolors.RESET)
                result = func(*args, **kwargs)
                print "{0} Total time taken by {1} for execution is --<<{2}>>-- from ip--{3}\n".format(bcolors.OKGREEN, func.func_name,
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
                                                        





def httpauth(arguments):
        def real_decorator(func):
                def wrapped(self, *args, **kwargs):
                        token =  self.get_argument("token")
                        print arguments
                        try:
                                header, claims = jwt.verify_jwt(token, public_key, ['RS256'])
                                self.claims = claims
                                self.messege = None
                                for __arg in arguments:
                                        try:
                                                claims[__arg]
                                        except Exception as e:
                                                self.messege = "Missing argument %s"%__arg
                                                self.set_status(400)
                        except _JWTError:
                                self.messege = "Token expired"
                                self.set_status(403)
                        except Exception as e:
                                self.messege = "Some error occurred"
                                print e
                                self.set_status(500)
                            
                        
                        if self.messege:
                                self.write({
                                        "error": True,
                                        "success": False, 
                                        "messege": self.messege, 
                                })
                                
                        return func(self, *args, **kwargs) 
                return wrapped                   
        return real_decorator


class GetKey(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                    if self.get_argument("secret") != '967d2b1f6111a198431532149879983a1ad3501224fb0dbf947499b1':
                            self.write({
                                "error": False,
                                "success": True, 
                                "messege": "Key, Nahi milegi", 
                                })
                            self.finish()
                            return 

                    self.write({
                                "error": True,
                                "success": False, 
                                "result": private, 
                        })
                    self.finish()
                    return 


class Test(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        @print_execution
        @httpauth(["latitude", "longitude"])
        def post(self):
                if not self.messege:
                        self.__on_response()
                self.finish()
                return 

        def __on_response(self):
                time.sleep(10)
                self.write({"success": True,
                            "error": False, 
                            "result": "success",
                            })
                return 

class GetApis(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                Args:
                        skip:
                        limit:
                        fb_id:

                result:
                    array of dicts
                            each object:
                                
                """
                if self.get_argument("key") != '967d2b1f6111a198431532149879983a1ad3501224fb0dbf947499b1':
                            self.write({
                                "error": False,
                                "success": True, 
                                "messege": "api, Nahi milegi", 
                                })
                            self.finish()
                            return 
                result = {
                        "suggestions": "{0}/{1}".format(server_address, "suggestions"), 
                        "textsearch": "{0}/{1}".format(server_address, "textsearch"), 
                        "getkey": "{0}/{1}".format(server_address, "getkey"), 
                        "userprofile": "{0}/{1}".format(server_address, "userprofile"), 
                        "gettrending": "{0}/{1}".format(server_address, "gettrending"), 
                        "nearesteateries": "{0}/{1}".format(server_address, "nearesteateries"), 
                        "usersdetails": "{0}/{1}".format(server_address, "usersdetails"), 
                        "usersfeedback": "{0}/{1}".format(server_address, "usersfeedback"), 
                        "writereview": "{0}/{1}".format(server_address, "writereview"), 
                        "fetchreview": "{0}/{1}".format(server_address, "fetchreview"), 
                        "geteatery": "{0}/{1}".format(server_address, "geteatery"),
                        }

                self.write({
                            "error": True, 
                            "success": False, 
                            "result": result,
                            })
                self.finish()
                return 

class UsersFeedback(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                x_real_ip = self.request.headers.get("X-Real-IP")
                remote_ip = x_real_ip or self.request.remote_ip
                feedback = self.get_argument("feedback")
                name = self.get_argument("name")
                email = self.get_argument("email")

                try:
                        fb_id = self.get_argument("fb_id")
                except Exception as e:
                        fb_id = None
                        pass
                
                if users_feedback_collection.find_one({"feedback": feedback, "name": name, "email": email, "ip": remote_ip}):
                        self.set_status(409)
                        self.write({
                            "error": True, 
                            "success": False, 
                            "messege": "This feedback has already been submitted"
                            })
                        self.finish()
                        return 
                
                users_feedback_collection.insert({"feedback": feedback, "fb_id": fb_id,  "epoch": time.time(), "name": name, "email": email, "ip": remote_ip})
                self.write({
                            "error": False, 
                            "success": True, 
                            "messege": "The feedback has been posted, Thank you for your efforts", 
                            })
                self.finish()
                return


class WriteReview(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                fb_id = self.get_argument("fb_id")
                review_text = self.get_argument("review_text")
                __eatery_id = self.get_argument("__eatery_id")
                __eatery_name = self.get_argument("eatery_name")


                if not users_details_collection.find_one({"fb_id": fb_id}):
                        self.set_status(401)
                        self.write({
                                "error": True, 
                                "success": False, 
                                "messege": "User needs to login"
                            })
                        self.finish()
                        return 


                fmt = "%d %B %Y %H:%M:%S"
                now_utc = datetime.now(timezone('UTC'))
                utctimestamp = __utc.strftime(fmt)


                indian = now_utc.astimezone(timezone('Asia/Kolkata'))
                indian_time = indian.strftime(fmt)

                user = users_details_collection.find_one({"fb_id": fb_id}, {"_id": False, "fb_id": False, "email": False})

                __dict = {"review_text": review_text, "fb_id": fb_id, "__eatery_id": __eatery_id, "__eatery_name": __eatery_name}
                review_id = hashlib.sha256(indian_time + __eatery_id + review_text + _dict.get("fb_id")).hexdigest()
                
                if users_reviews_collection.find_one(__dict):
                        self.write({
                            "error": True, 
                            "success": False, 
                            "messege": "This review has already been posted"
                            })
                        self.finish()
                        return 
                
                __dict.update({"utc": utctimestamp})
                __dict.update({"epoch": indian_time })
                __dict.update(user)
                __dict.update({"review_id": review_id})
                users_reviews_collection.insert(__dict)
                
                self.write({
                            "error": False, 
                            "success": True, 
                            "messege": "This review has been posted",
                            })
                self.finish()
                return 

            
class FetchReview(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                __eatery_id = self.get_argument("__eatery_id")
                try:
                        limit = int(self.get_argument("limit"))
                except Exception as e:
                        print e
                        limit = 10
                
                try:
                        skip = int(self.get_argument("skip"))
                except Exception as e:
                        print e
                        skip = 0
                

                result = list()
                
                if not users_reviews_collection.find_one({"__eatery_id": __eatery_id}):
                        self.write({
                                    "error": True,
                                    "success": False, 
                                    "messege": "No reviews are present",
                                })
                        self.finish()
                        return 
                        

                for review in users_reviews_collection.find({"__eatery_id": __eatery_id}, {"_id": False, "utc": False}).sort("epoch", -1).skip(skip).limit(limit):
                        result.append(review)

                self.write({
                            "error": False, 
                            "success": True, 
                            "result": result,
                            })
                self.finish()
                return 



class UsersDetails(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                """
                User when does a fb login
                """
                fb_id = self.get_argument("fb_id")
                name = self.get_argument("name")
                email = self.get_argument("email")
                picture = self.get_argument("picture")
                print users_details_collection.update({"fb_id": fb_id}, {"$set": { "name": name, "email": email, "picture": picture}}, upsert=True, multi=False)
                self.write({"success": True,
			"error": False,
			})
                self.finish()
                return



class GetTrending(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                Returns 10 nearest entries based on the four categories  to the latitude and longitude given to it
                Args:
                    latitude
                    longitude
                """
                latitude = float(self.get_argument("latitude"))
                longitude = float(self.get_argument("longitude"))
                print type(longitude)
                __result = ElasticSearchScripts.get_trending(latitude, longitude)
                

                print __result["service"]
                result = dict()
                for __category in [u'food', u'ambience', u'cost', u'service']:
                        __list = list()
                        for element in __result[__category]:
                                __eatery_id = element.get("__eatery_id")
                                __eatery_details = short_eatery_result_collection.find_one({"__eatery_id": __eatery_id})
                                for e in ["eatery_highlights", "eatery_cuisine", "eatery_trending", "eatery_id", "eatery_known_for", "eatery_type", "_id"]:
                                        try:
                                                __eatery_details.pop(e)
                                        except Exception as e:
                                                pass
                                element.update({"eatery_details": __eatery_details})
                                __list.append(element)
                        
                        result[__category] = __list

                self.write({"success": True,
			        "error": False,
			        "result": result,
			        })
                self.finish()
                return 


##edited on 24 december Done
class NearestEateries(tornado.web.RequestHandler):
	@cors
	@print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def post(self):
                """
                Accoriding to the latitude, longitude given to it gives out the 10 restaurants nearby
                """
                
                latitude =  float(self.get_argument("latitude"))
                longitude =  float(self.get_argument("longitude")) 
                

                #result = eateries.find({"eatery_coordinates": {"$near": [lat, long]}}, projection).sort("eatery_total_reviews", -1).limit(10)
                #result = eateries.find({"eatery_coordinates" : SON([("$near", { "$geometry" : SON([("type", "Point"), ("coordinates", [lat, long]), \
                #        ("$maxDistance", range)])})])}, projection).limit(10)


                try:
                        short_eatery_result_collection.index_information()["location_2d"]

                except Exception as e:
                        self.write({"success": False,
			        "error": True,
                                "result": "Location index not present of collection",
			    })
                        self.finish()
                        return 
                        
                projection={"__eatery_id": True, "eatery_name": True, "eatery_address": True, "location": True, "_id": False, "food": True, \
                        "overall": True}
                
                result = short_eatery_result_collection.find({"location": { "$geoWithin": { "$centerSphere": [[latitude, longitude], .5/3963.2] } }}, \
                        projection).sort("overall.total_sentiments", -1).limit(10)
                ##__result  = list(result)

                final_result = list()
                for element in result:
                            sentiments = element.pop("overall")
                            dishes = element.pop("food")
                            element.update({"eatery_details": 
                                {"location": element.pop("location"),
                                    "__eatery_id": element.get("__eatery_id"), 
                                    "eatery_address": element.pop("eatery_address"), 
                                    "eatery_name": element.pop("eatery_name"),
                                    "overall": {"total_sentiments": sentiments.get("total_sentiments")},
                                    "food": dishes,
                                    }})
                                    
                            element.update({"excellent": sentiments.get("excellent"), 
                                    "poor": sentiments.get("poor"), 
                                    "good": sentiments.get("good"), 
                                    "average": sentiments.get("average"), 
                                    "terrible": sentiments.get("terrible"), 
                                    "total_sentiments": sentiments.get("total_sentiments"),    
                                    })

                            final_result.append(element)


                final_result = sorted(final_result, reverse=True, key = lambda x: x.get("total_sentiments"))
                self.write({"success": True,
			"error": False,
                        "result": final_result,
			})
                self.finish()
                return 
                

class TextSearch(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                This api will be called when a user selects or enter a query in search box
                """
                text = self.get_argument("text")
                __type = self.get_argument("type")


                if __type == "dish":
                        """
                        It might be a possibility that the user enetered the dish which wasnt in autocomplete
                        then we have to search exact dish name of seach on Laveneshtein algo
                        """
                        ##search in ES for dish name 

                        result = list()
                        __result = ElasticSearchScripts.get_dish_match(text)
                        for dish in __result:
                                __eatery_id = dish.get("__eatery_id")
                                __eatery_details = short_eatery_result_collection.find_one({"__eatery_id": __eatery_id})
                                for e in ["eatery_highlights", "eatery_cuisine", "eatery_trending", "eatery_id", "eatery_known_for", "eatery_type", "_id"]:
                                        try:
                                                __eatery_details.pop(e)
                                        except Exception as e:
                                                pass
                                dish.update({"eatery_details": __eatery_details})
                                result.append(dish)

                elif __type == "cuisine":
                        ##gives out the restarant for cuisine name
                        print "searching for cuisine"
                        result = list()
                        __result = ElasticSearchScripts.eatery_on_cuisines(text)
                        print __result
                        for eatery in __result:
                                    __result= short_eatery_result_collection.find_one({"__eatery_id": eatery.get("__eatery_id")}, {"_id": False, "food": True, "ambience": True, \
                                            "cost":True, "service": True, "menu": True, "overall": True, "location": True, "eatery_address": True, "eatery_name": True, "__eatery_id": True})

                                    eatery.update({"eatery_details": __result})
                                    result.append(eatery)

                elif __type == "eatery":
                       
                            
                            result = eateries_results_collection.find_one({"eatery_name": text})
                            __result = process_result(result)
                            for e in ["eatery_highlights", "eatery_cuisine", "eatery_trending", "eatery_id", "eatery_known_for", "eatery_type", "_id", "processed_reviews", "old_considered_ids"]:
                                    try:
                                            result.pop(e)
                                    except Exception as e:
                                            pass
                            
                            result.update(__result)


                elif not  __type:
                        print "No type defined"

                else:
                        print __type
                        self.write({"success": False,
			        "error": True,
			        "messege": "Maaf kijiyega, Yeh na ho paayega",
			        })
                        self.finish()
                        return 
                self.write({"success": False,
			        "error": True,
			        "result": result,
			})
                self.finish()
                return 



class Suggestions(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """


                Return:

                        [
                        {u'suggestions': [u'italian food', 'italia salad', 'italian twist', 'italian folks', 'italian attraction'], 'type': u'dish'},
                        {u'suggestions': [{u'eatery_name': u'Slice of Italy'}], u'type': u'eatery'},
                        {u'suggestions': [{u'name': u'Italian'}, {u'name': u'Cuisines:Italian'}], 'type': u'cuisine'}
                        ]

                """
                        
                query = self.get_argument("query")
                
                dish_suggestions = ElasticSearchScripts.dish_suggestions(query)
                cuisines_suggestions =  ElasticSearchScripts.cuisines_suggestions(query)
                eatery_suggestions = ElasticSearchScripts.eatery_suggestions(query)
                #address_suggestion = ElasticSearchScripts.address_suggestions(query)
                

                if cuisines_suggestions:
                        cuisines_suggestions= [e.get("name") for e in cuisines_suggestions]
                
                if eatery_suggestions:
                        eatery_suggestions= [e.get("eatery_name") for e in eatery_suggestions]


                self.write({"success": True,
			        "error": False,
                                "result": [{"type": "dish", "suggestions": [e.get("name") for e in dish_suggestions] },
                                            {"type": "eatery", "suggestions": eatery_suggestions },
                                            {"type": "cuisine", "suggestions": cuisines_suggestions }
                                            ]
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
                        
                __eatery_id =  self.get_argument("__eatery_id")
                result = eateries_results_collection.find_one({"__eatery_id": __eatery_id})
                images = list(pictures_collection.find({"__eatery_id": __eatery_id}, {"_id": False, "s3_url": True, "image_id": True, "likes": True}).limit(10))
                if not result:
                        """
                        If the eatery name couldnt found in the mongodb for the popular matches
                        Then we are going to check for demarau levenshetin algorithm for string similarity
                        """

                        self.write({"success": False,
			        "error": True,
                                "result": "Somehow eatery with this eatery is not present in the DB"})
                        self.finish()
                        return 
               

                cprint(figlet_format('Finished executing %s'%self.__class__.__name__, font='mini'), attrs=['bold'])
                
                __result = process_result(result)
                __result.update({"images": images})
                self.write({"success": True,
			"error": False,
                        "result": __result})
                self.finish()


                return 



def process_google_result(result, __eatery_id):
                """

                """
                google = eateries_results_collection.find_one({"__eatery_id": __eatery_id}).get("google")
                try:
                        google = result.pop("google")
                        if google == 'No entry exists on google':
                                return result
                except Exception as e:
        
                        pass

                if google.get("location"):
                        result["location"] = google.get("location")
                        print "location found in google result for %s"%__eatery_id

                if google.get("eatery_address"):
                        result["eatery_address"] = google.get("eatery_address")
                        print "Address found in google result for %s"%__eatery_id
                
                if google.get("eatery_phone_number"):
                        result["eatery_phone_number"] = google.get("eatery_phone_number")
                else:
                        result["eatery_phone_number"] = None
                        

                return result



class GetUserProfile(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                """
                fb_id = self.get_argument("fb_id")

                try:
                        limit = self.get_argument("limit")
                except Exception as e:
                        print e
                        limit = 10
                
                try:
                        skip = self.get_argument("skip")
                except Exception as e:
                        print e
                        skip = 0



                __result =  [post for post in users_reviews_collection.find({"fb_id": fb_id}).skip(skip).limit(limit)]
                if not __result:
                        self.write({"success": False,
			        "error": True, 
                                "messege": "No reviews present for this user", 
                                })
                        self.finish()
                        return 

                [post.pop("_id") for post in __result]
                self.write({"success": True,
			        "error": False,
                                "result": __result,  
                                })
                
                
                self.finish()
                return 





class UploadPic(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                This api end point will be used to upload picture for the eatery, also 
                user can upload pic for the related dish, in that case a dish name has be made available
                the picture that has been uploaded successfult will have following fields 

                    "epoch": indian_time, 
                    "utctimestamp": utctimestamp, 
                    "pic_id": pic_id, 
                    "data": data, 
                    "__eatery_id": __eatery_id, 
                    "fb_id": fb_id, 
                    "likes": Int number of likes
                    "users": "A list of users with their fb_ids and their picture url who has liked this pic
                """

                __eatery_id = self.get_argument("__eatery_id")
                pic_data = self.get_argument("pic_data")
                fb_id = self.get_argument("fb_id")

                picture = {}
                try:
                        dish_name = selg.get_argument("dish_name")
                except Exception as e:
                        dish_name = None


                try:
                        based.b64decode(pic_data)

                except TypeError as e:
                        self.write({"success": False,
			        "error": True,
                                "messege": "Image cant be uploaded because of the incorrect format",  
                                })
                        self.finish()
                        return 
                        
                fmt = "%d %B %Y %H:%M:%S"
                now_utc = datetime.now(timezone('UTC'))
                utctimestamp = __utc.strftime(fmt)


                indian = now_utc.astimezone(timezone('Asia/Kolkata'))
                indian_time = indian.strftime(fmt)

                picture.update({"epoch": indian_time, "utctimestamp": utctimestamp, "pic_id": pic_id, "data": data, "__eatery_id": __eatery_id, "fb_id": fb_id, "likes": 0, "users": []})

                try:
                        picture_collection.insert(picture)
                        success = True
                        error = False
                        messege = "The image has been uploaded successfully"
                except Exception as e:
                        success = False
                        error =True
                        messege = "The image already exists"

                self.write({"success": success,
			        "error": error,
                                "messege": messege,  
                                })
                self.finish()
                return 



        
                
class LikePic(tornado.web.RequestHandler):
        """
        If a user wants to like a pic, for that to happend a pic_id and fb_id must be given
        a like counter shall be incremented by one if the user has alread not have likened the pic
        and alos users_details_collection shall be update with s3_url
        """
        @cors
	@tornado.gen.coroutine
        def post(self):
                print "lihe_pic called"
                image_id = self.get_argument("image_id")
                print image_id
                fb_id = self.get_argument("fb_id")
                print fb_id
                s3_url = self.get_argument("s3_url")
                print s3_url

                print image_id, fb_id, s3_url
                if not users_details_collection.find_one({"fb_id": fb_id}):
                        self.write({"success": False,
			        "error": True,
                                "messege": "User havent been registered, Please login with your facebook",  
                                })
                        self.finish()
                        return 
                
                
                
                
                
                if pictures_collection.find_one({"image_id": image_id, "users": {"fb_id": fb_id}}):
                        self.write({"success": False,
			        "error": True,
                                "messege": "User already have liked the picture",  
                                })
                        self.finish()
                        return 

                        


                pictures_collection.update({"image_id": image_id}, {"$inc": {"likes": 1}, "$addToSet": {"users": {"fb_id": fb_id}}},  upsert=False)
                users_details_collection.update({"fb_id": fb_id}, {"$addToSet": {"images": {"s3_url": s3_url, "image_id": image_id}}},  upsert=False)
                self.write({"success": True,
			    "error": False,
                            "messege": "Picture has been liked by the user",  
                            })
                self.finish()
                return 




                
class LikeReview(tornado.web.RequestHandler):
        """
        If a user wants to like a review, for that to happened a pic_id and fb_id must be given
        """
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                review_id = self.get_argument("review_id")
                fb_id = self.get_argument("fb_id")

                if users_reviews_collection.find_one({"review_id": review_id, "users": {"fb_id": fb_id}}):
                        self.write({"success": False,
			        "error": True,
                                "messege": "User already have liked the picture",  
                                })
                        self.finish()
                        return 

                        



                users_reviews_collection.update({"review_id": review_id}, {"$inc": {"likes": 1}, "$addToSet": {"users": {"fb_id": fb_id}}},  upsert=False)
                users_details_collection.update({"fb_id": fb_id}, {"$addToSet": {"reviews_liked": {"reviews_id": review_id}}},  upsert=False)
                
                self.write({"success": True,
			    "error": False,
                            "messege": "Review has been liked by the user",  
                            })
                self.finish()
                return 




class StoreAddressStrings(tornado.web.RequestHandler):
        """
        When a user enters the address query or when preseeed enter that address is to be pushed at this end point
        this will be helpful for us later
        One example could be to provde autocomlpete address strings rather then using google places api
        """
        @cors
	@print_execution
	@tornado.gen.coroutine
        def  post(self):
                address_string = self.get_argument("address_string")
                try:
                        fb_id = self.get_argument("fb_id")
                except Exception as e:
                        fb_id = None

                x_real_ip = self.request.headers.get("X-Real-IP")
                remote_ip = x_real_ip or self.request.remote_ip

                users_queried_addresses_collection.insert({"address_string": address_string, "fb_id": fb_id, "ip": remote_ip, "epoch": time.time()})
                self.write({
                        "success": True, 
                        "error": False, 
                        })




class AddDish(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                If a user wants to add dish to an eatery_id, This will be stored in a different collection called as user_dish_collection
                A user also has to mention a sentiment, whether excellent, good, poor, average, terrible
                
                """
                dish = self.get_argument("dish")
                fb_id = self.get_argument("fb_id")

                __eatery_id = self.get_argument("__eatery_id")
                sentiment = self.get_argument("sentiment")


                dish_id = hashlib.sha256(__eatery_id + dish).hexdigest()
                     
                if sentiment not in ["terrible", "average", "good", "excellent", "poor"]:
                        self.write({ "success": False, 
                                    "error": True, 
                                    "messege": "The sentiment category that the user has defined is not allowed",
                            })
        

                if user_dish_collection.find_one({"eatery_id": eatery_id}, {"$exists": {dish: True}}):
                        ##if a dish is already present the sentiment will be added to that dish 
                        user_dish_collection.update({"eatery_id": eatery_id}, {"$addToSet": {dish: sentiment }})
                        self.write({"success": False,
			        "error": True, 
                                "messege": "The dish already exists for the eatery", 
                                })
                        self.finish()
                        return 
                
                ##upsert=False, becuase the eatery already exists in the database




def process_result(result):
                number_of_dishes = 20
                dishes = sorted(result["food"]["dishes"], key=lambda x: x.get("total_sentiments"), reverse=True)[0: number_of_dishes]
                overall_food = result["food"]["overall-food"]
                ambience = result["ambience"]
                cost = result["cost"]
                service = result["service"]
                overall = result["overall"]
                menu = result["menu"]

                ##removing timeline
                [value.pop("timeline") for (key, value) in ambience.iteritems()]
                [value.pop("timeline") for (key, value) in cost.iteritems()]
                [value.pop("timeline") for (key, value) in service.iteritems()]
                overall.pop("timeline")
                menu.pop("timeline")
                [element.pop("timeline") for element in dishes]
                [element.pop("similar") for element in dishes]



                result = {"food": dishes,
                            "ambience": ambience, 
                            "cost": cost, 
                            "service": service, 
                            "menu": menu,
                            "overall": overall,
                            }
                        
                return result


app = tornado.web.Application([
                    (r"/suggestions", Suggestions),
                    (r"/textsearch", TextSearch),
                    (r"/test", Test),
                    (r"/getkey", GetKey),
                    (r"/userprofile", GetUserProfile),
                    (r"/apis", GetApis),
                    (r"/likepic", LikePic),
                    (r"/likereview", LikeReview),
                    (r"/storeaddressstrings", StoreAddressStrings),
                                        
                    (r"/gettrending", GetTrending),
                    (r"/nearesteateries", NearestEateries),
                    (r"/usersdetails", UsersDetails),
                    (r"/usersfeedback", UsersFeedback),
                    (r"/writereview", WriteReview),
                    (r"/fetchreview", FetchReview),
                    (r"/geteatery", GetEatery),])

def main():
        http_server = tornado.httpserver.HTTPServer(app)
        """
        http_server.listen("8000")
        enable_pretty_logging()
        tornado.ioloop.IOLoop.current().start()
        """
        http_server.bind("8000")
        enable_pretty_logging()
        http_server.start(0) 
        loop = tornado.ioloop.IOLoop.instance()
        loop.start()
"""
class Application(tornado.web.Application):
        def __init__(self):
                handlers = [
                    (r"/suggestions", Suggestions),
                    (r"/textsearch", TextSearch),
                    
                    (r"/gettrending", GetTrending),
                    (r"/nearesteateries", NearestEateries),
                    (r"/usersdetails", UsersDetails),
                    (r"/usersfeedback", UsersFeedback),
                    (r"/geteatery", GetEatery),]
                tornado.web.Application.__init__(self, handlers, **settings)
                self.executor = ThreadPoolExecutor(max_workers=60)

"""

if __name__ == '__main__':
    cprint(figlet_format('Server Reloaded', font='big'), attrs=['bold'])
    main()
