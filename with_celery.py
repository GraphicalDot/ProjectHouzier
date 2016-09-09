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
from flask import Flask
from flask import request, jsonify
from flask.ext import restful
from flask.ext.restful import reqparse
from flask import make_response, request, current_app
from functools import update_wrapper
from flask import jsonify
import hashlib
import subprocess
import shutil
import json
import os
import StringIO
import difflib
from bson.json_util import dumps
from Text_Processing import NounPhrases, get_all_algorithms_result, RpRcClassifier, \
		bcolors, CopiedSentenceTokenizer, SentenceTokenizationOnRegexOnInterjections, get_all_algorithms_result, \
		path_parent_dir, path_trainers_file, path_in_memory_classifiers, timeit, cd, SentimentClassifier, \
		TagClassifier, NERs, NpClustering 

from compiler.ast import flatten
##TODO run check_if_hunpos and check_if stanford fruntions for postagging and NERs and postagging


from Text_Processing import WordTokenize, PosTaggers, NounPhrases


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
from ProcessingCeleryTask import MappingList, SentTokenizeToNP, ReviewIdToSentTokenize, CleanResultBackEnd,\
                    Clustering, StoreInEatery, NoNounPhrasesReviews, sub_categories
from celery.result import AsyncResult
from celery import chord
from heuristic_clustering import HeuristicClustering

from GlobalConfigs import MONGO_REVIEWS_IP, MONGO_REVIEWS_PORT, MONGO_REVIEWS_DB,\
        MONGO_REVIEWS_EATERIES_COLLECTION, MONGO_REVIEWS_REVIEWS_COLLECTION

connection = pymongo.MongoClient(MONGO_REVIEWS_IP, MONGO_REVIEWS_PORT, tz_aware=True, w=1, 
        j=True, max_pool_size=200, use_greenlets=True)


eateries = eval("connection.{db_name}.{collection_name}".format(
                                                        db_name=MONGO_REVIEWS_DB,
                                                        collection_name=MONGO_REVIEWS_EATERIES_COLLECTION))


reviews = eval("connection.{db_name}.{collection_name}".format(
                                    db_name=MONGO_REVIEWS_DB,
                                    collection_name=MONGO_REVIEWS_REVIEWS_COLLECTION))




training_db = connection.training_data
training_sentiment_collection = training_db.training_sentiment_collection
training_tag_collection = training_db.training_tag_collection

#This is for android apps, may not be required later
android_db = connection.android_app
android_users = android_db.users
users_pic = android_db.pics
#####


app = Flask(__name__)
app.config['DEBUG'] = True
api = restful.Api(app,)





def to_unicode_or_bust(obj, encoding='utf-8'):
	if isinstance(obj, basestring):
		if not isinstance(obj, unicode):
			obj = unicode(obj, encoding)
	return obj



def word_tokenization_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(WordTokenize, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]
        
        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for word tokenization doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name

def pos_tagging_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(PosTaggers, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for Pos Tagging  doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name

def noun_phrases_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NounPhrases, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for noun phrases doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name




def tag_analysis_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(TagClassifier, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for tag analysis doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name

def sentiment_analysis_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(SentimentClassifier, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for sentiment analysis doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name

def np_clustering_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NpClustering, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__",]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for noun phrase clustering doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name


def ner_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NERs, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__",]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for ner extraction doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name


def custom_string(__str):
        return __str.encode("utf-8")


#fb_login
fb_login_parser = reqparse.RequestParser()
fb_login_parser.add_argument("fb_id", type=str, required=True, location="form")
fb_login_parser.add_argument("email", type=str, required=False, location="form")
fb_login_parser.add_argument("gender", type=str, required=True, location="form")
fb_login_parser.add_argument("user_name", type=str, required=True, location="form")
fb_login_parser.add_argument("date_of_birth", type=str, required=True, location="form")
fb_login_parser.add_argument("location", type=str, required=True, location="form")
fb_login_parser.add_argument("user_friends", type=str, required=False, location="form", action="append")



#post_comment
post_comment_parser = reqparse.RequestParser()
post_comment_parser.add_argument("fb_id", type=str, required=True, location="form")
post_comment_parser.add_argument("comment", type=str, required=True, location="form")


#suggest_name
suggest_name_parser = reqparse.RequestParser()
suggest_name_parser.add_argument("fb_id", type=str, required=True, location="form")
suggest_name_parser.add_argument("name_suggestion", type=str, required=True, location="form")


#post_picture
post_pic_parser = reqparse.RequestParser()
post_pic_parser.add_argument("fb_id", type=str, required=True, location="form")
post_pic_parser.add_argument("image_name", type=str, required=True, location="form")
post_pic_parser.add_argument("image_string", type=str, required=True, location="form")


#get_pics
get_pics_parser = reqparse.RequestParser()
get_pics_parser.add_argument("dish_name", type=str, required=True, location="args")


##GetStartDateForRestaurant
get_start_date_for_restaurant_parser = reqparse.RequestParser()
get_start_date_for_restaurant_parser.add_argument('eatery_id', type=str,  required=True, location="form")
word_cloud_with_dates_parser = reqparse.RequestParser() 


##GetWordCloud
get_word_cloud_parser = reqparse.RequestParser()
get_word_cloud_parser.add_argument('eatery_id', type=str,  required=True, location="form")
get_word_cloud_parser.add_argument('category', type=str,  required=True, location="form")
get_word_cloud_parser.add_argument('start_date', type=str,  required=False, location="form")
get_word_cloud_parser.add_argument('end_date', type=str,  required=False, location="form") 
get_word_cloud_parser.add_argument('total_noun_phrases', type=int,  required=False, location="form") 
get_word_cloud_parser.add_argument('word_tokenization_algorithm', type=word_tokenization_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('noun_phrases_algorithm', type=noun_phrases_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('pos_tagging_algorithm', type=pos_tagging_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('tag_analysis_algorithm', type=tag_analysis_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('sentiment_analysis_algorithm', type=sentiment_analysis_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('np_clustering_algorithm', type=np_clustering_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('ner_algorithm', type=ner_algorithm,  required=False, location="form")


##raw_text_processing_parser
raw_text_processing_parser = reqparse.RequestParser()
raw_text_processing_parser.add_argument('text', type=custom_string,  required=True, location="form")


change_tag_or_sentiment_parser = reqparse.RequestParser()
change_tag_or_sentiment_parser.add_argument('sentence', type=str,  required=True, location="form")
change_tag_or_sentiment_parser.add_argument('value', type=str,  required=True, location="form")
change_tag_or_sentiment_parser.add_argument('whether_allowed', type=str,  required=False, location="form")

def cors(func, allow_origin=None, allow_headers=None, max_age=None):
	if not allow_origin:
                allow_origin = "*"
                		
	if not allow_headers:
		allow_headers = "content-type, accept"
		
	if not max_age:
		max_age = 60

	@wraps(func)
	def wrapper(*args, **kwargs):
		response = func(*args, **kwargs)
		cors_headers = {
				"Access-Control-Allow-Origin": allow_origin,
				"Access-Control-Allow-Methods": func.__name__.upper(),
				"Access-Control-Allow-Headers": allow_headers,
				"Access-Control-Max-Age": max_age,
				}
		if isinstance(response, tuple):
			if len(response) == 3:
				headers = response[-1]
			else:
				headers = {}
			headers.update(cors_headers)
			return (response[0], response[1], headers)
		else:
			return response, 200, cors_headers
	return wrapper


class FBLogin(restful.Resource):
	@cors
	def post(self):
                """
                If the length of the new user_friends pposted ont he api uis greater than the length
                of the user_friends present in the database,
                then the list of user_friends shall be updated in the database

                """

		args = fb_login_parser.parse_args()
                print args
                if not android_users.find_one({"fb_id": args["fb_id"]}):
                        android_users.update({"fb_id": args["fb_id"]}, {"$set": {
                                                "user_name": args["user_name"],
                                                "email": args["email"],
                                                "gender": args["gender"], 
                                                "date_of_birth": args["date_of_birth"],
                                                "location": args["location"],
                                                "user_friends": args["user_friends"],}} , upsert=True) 
                
                        return {"error": False,
                                "success": True,
                                "error_code": 0,
                                "messege": "The user with fb_id {0} and name {1} has been inserted correctly".
                                                    format(args["fb_id"], args["user_name"] ),}
                
                
                if android_users.find_one({"fb_id": args["fb_id"]}):
                        if len(android_users.find_one({"fb_id": args["fb_id"]}).get("user_friends")) < len(args["user_friends"]):
                                android_users.update({"fb_id": args["fb_id"]}, {"$set": {
                                                "user_friends": args["user_friends"],}} , upsert=False) 
                                
                    
                                return {"error": False,
                                        "success": True,
                                        "error_code": 0,
                                        "messege": "The user with fb_id {0} and name {1} has been updated with new user_friends".
                                                    format(args["fb_id"], args["user_name"] ),}
                
                        return {"error": True,
                                "success": False,
                                "error_code": 0, 
                                "messege": "The user with fb_id {0} and name {1} already exists".
                                                    format(args["fb_id"], args["user_name"] ),}
                
                return


class PostComment(restful.Resource):
	@cors
	def post(self):
                args = post_comment_parser.parse_args()
                if not android_users.find_one({"fb_id": args["fb_id"]}):
                        return {"error": True,
                                "success": False,
                                "messege": "Please register the user first before posting the comment",}
                
                android_users.update({"fb_id": args["fb_id"]}, {"$push": {
                                                "comments": args["comment"],}}) 
            
                return {"error": False,
                        "success": True,
                        "messege": "The comment has been posted successfully",}
                        



class SuggestName(restful.Resource):
	@cors
	def post(self):
                args = suggest_name_parser.parse_args()
                if not android_users.find_one({"fb_id": args["fb_id"]}):
                        return {"error": True,
                                "success": False,
                                "messege": "Please register the user first before posting the comment",}
                
                android_users.update({"fb_id": args["fb_id"]}, {"$push": {
                                                "name_suggestion": args["name_suggestion"],}}) 
            
                return {"error": False,
                        "success": True,
                        "messege": "The suggestion for the name has been taken successfully",}



            
class PostPicture(restful.Resource):
	@cors
	def post(self):
                args = post_pic_parser.parse_args()
                if not android_users.find_one({"fb_id": args["fb_id"]}):
                        return {"error": True,
                                "success": False,
                                "messege": "Please register the user first before posting the image",}

                try:
                        base64.decodestring(args["image_string"])

                except Exception as e:
                        return {"error": True,
                                "success": False,
                                "messege": "The pic cannot be posted because of the error {0}".format(e),}

                
                #md5 checksum of the base64 encoded image, to form its unique id
                image_id = hashlib.md5(args["image_string"]).hexdigest() 


                #to check whether the same user is going to upload the same pic again
                if users_pic.find_one({"fb_id": args["fb_id"], "image_id": image_id }):
                        return {"error": True,
                                "success": False,
                                "messege": "This pic for this user has already been posted",}
                        


                android_users.update({"fb_id": args["fb_id"]}, {"$push": {
                                            "pics": image_id,}}) 
                
                users_pic.insert({"image_id": image_id, "image_name": args["image_name"], "fb_id": args["fb_id"], 
                                "image_base64_encoded_string": args["image_string"]})


                return {"error": False,
                        "success": True,
                        "messege": "The pic has been posted successfully",}


                #do we have image name options to be selected from
            
class GetPics(restful.Resource):
	@cors
	def get(self):
                args = get_pics_parser.parse_args()
                args["dish_name"]
                return {"error": False,
                        "success": True,
                        "result": result,}








class LimitedEateriesList(restful.Resource):
	@cors
	def get(self):
                """
                This gives only the limited eatery list like the top on the basis of the reviews count
                """
                result = list(eateries.find({"eatery_area_or_city": "ncr"}, fields= {"eatery_id": True, "_id": False, "eatery_name": True}).limit(200).sort("eatery_total_reviews", -1))
		
                return {"success": True,
			"error": False,
			"result": result,
			}

                
class EateriesList(restful.Resource):
	@cors
	def get(self):
                result = list(eateries.find({"eatery_area_or_city": "ncr"}, fields= {"eatery_id": True, "_id": False, "eatery_name": True}).limit(200).sort("eatery_total_reviews", -1))
		
                return {"success": True,
			"error": False,
			"result": result,
			}



class GetStartDateForRestaurant(restful.Resource):
	@cors
	def post(self):
		args = get_start_date_for_restaurant_parser.parse_args()
		eatery_id = args["eatery_id"]
		sorted_list_by_epoch = list(reviews.find({"eatery_id" :eatery_id}).sort("converted_epoch", 1))
		start_date = sorted_list_by_epoch[0].get('readable_review_day')
		start_month = sorted_list_by_epoch[0].get('readable_review_month')
		start_year = sorted_list_by_epoch[0].get('readable_review_year')
		
		end_date = sorted_list_by_epoch[-1].get('readable_review_day')
		end_month = sorted_list_by_epoch[-1].get('readable_review_month')
		end_year = sorted_list_by_epoch[-1].get('readable_review_year')
		
		
		
		
		return {"success": True,
				"error": True,
				"result": {"start": "{0}-{1}-{2}".format(start_year, start_month, start_date), 
					"end": "{0}-{1}-{2}".format(end_year, end_month, end_date)},
		}


class GetWordCloud(restful.Resource):
	@cors
	@timeit
        def post(self):
		"""
                __test_eateries = [("7227", 232), ("4114", 50), ("154", 25), ("307799", 109), ("4815", 403), ("94286", 704)]
                To test
                    eatery_id = "4571"
                    start_epoch = 1318185000.0
                    end_epoch = 1420569000.0
                    
                    start_date = "2011-10-10"
                    end_date = "2015-01-07"
                    category = "food"
                """ 
                args = get_word_cloud_parser.parse_args()
                start = time.time()
		__format = '%Y-%m-%d'
		eatery_id = args["eatery_id"]
		category = args["category"].lower()
                
                total_noun_phrases = (None, args["total_noun_phrases"])[args["total_noun_phrases"] != None]
                
                word_tokenization_algorithm_name = ("punkt_n_treebank", args["word_tokenization_algorithm"])[args["word_tokenization_algorithm"] != None]
                
                
                noun_phrases_algorithm_name = ("textblob_np_conll", args["noun_phrases_algorithm"])[args["noun_phrases_algorithm"] != None]
                
                
                pos_tagging_algorithm_name = ("hunpos_pos_tagger", args["pos_tagging_algorithm"])[args["pos_tagging_algorithm"] != None]
                
                tag_analysis_algorithm_name = ("svm_linear_kernel_classifier_tag.lib", 
                                                    "{0}_tag.lib".format(args["tag_analysis_algorithm"]))[args["tag_analysis_algorithm"] != None]
                
                sentiment_analysis_algorithm_name = ("svm_linear_kernel_classifier_sentiment.lib", 
                                    "{0}_sentiment.lib".format(args["sentiment_analysis_algorithm"]))[args["sentiment_analysis_algorithm"] != None]

                np_clustering_algorithm_name = ("k_means", args["np_clustering_algorithm"])[args["np_clustering_algorithm"] != None]
                
                ner_algorithm_name = ("nltk_maxent_ner", args["ner_algorithm"])[args["ner_algorithm"] != None]
    
                if args["start_date"] and ["end_date"]:
		        try:
		                start_epoch = time.mktime(time.strptime(args["start_date"], __format))
			        end_epoch = time.mktime(time.strptime(args["end_date"], __format))
		        except Exception:
			        return {"success": False,
                                        "error": True,
                                        }

                else:
                        start_epoch, end_epoch = None, None
	

                if tag_analysis_algorithm_name.replace("_tag.lib", "") != sentiment_analysis_algorithm_name.replace("_sentiment.lib", ""):
                        return {"error": True,
                                "success": False,
                                "error_messege": "Right now, only the same algortihm can be used for tag, sentiment and cost analysis,\
                                        Make sure you are sending the same algortihm name for all the classification problems", 
                                }
                        




		print "start epoch is -->%s and end_apoch is -->%s"%(start_epoch, end_epoch) 
                if not reviews.find({"eatery_id": eatery_id}):
                        return {"error": True,
                                "success": False,
                                "error_messege": "The eatery id  {0} is not present".format(eatery_id), 
                                }



                #name of the eatery
                eatery_name = eateries.find_one({"eatery_id": eatery_id}).get("eatery_name")

                if category not in ["service", "food", "ambience", "null", "overall", "cost"]:
                        return {"error": True,
                                "success": False,
                                "error_messege": "This is a n invalid tag %s"%category, 
                                }
        
                

                print eatery_id, category, start_epoch,  end_epoch, tag_analysis_algorithm_name, sentiment_analysis_algorithm_name,\
                        word_tokenization_algorithm_name, pos_tagging_algorithm_name, noun_phrases_algorithm_name, np_clustering_algorithm_name,\
                        total_noun_phrases, ner_algorithm_name
               
                result = list()
      
                if category == "cost":
                        celery_chain = ReviewIdToSentTokenize.apply_async(args=[eatery_id, category, start_epoch, 
                            end_epoch, tag_analysis_algorithm_name, sentiment_analysis_algorithm_name, 
                            word_tokenization_algorithm_name, pos_tagging_algorithm_name, noun_phrases_algorithm_name])


                       

                        __sentences = sub_categories(eatery_id, category, start_epoch, end_epoch, tag_analysis_algorithm_name)
                        review_ids, sentences = zip(*[(__e[0], __e[1]) for __e in __sentences])
                        
                        
                        file_path = os.path.dirname(os.path.abspath(__file__))
                        classifier_path = "{0}/Text_Processing/PrepareClassifiers/InMemoryClassifiers/".format(file_path)
                        classifier = joblib.load("{0}{1}".format(classifier_path, "svm_linear_kernel_classifier_cost.lib"))
                        
                        for k, v in Counter(classifier.predict(sentences)).items():
                                if k == "cost-null":
                                        pass
                                else:
                                    result.append({"name": k,
                                                    "positive": v,
                                                    "negative": 0})
                        return {"success": True,
				"error": False,
                                "result": result,
                                }
                """
                if category == "ambience":
                        celery_chain = ReviewIdToSentTokenize.apply_async(args=[eatery_id, category, start_epoch, 
                            end_epoch, tag_analysis_algorithm_name, sentiment_analysis_algorithm_name, 
                            word_tokenization_algorithm_name, pos_tagging_algorithm_name, noun_phrases_algorithm_name])


                        __sentences = sub_categories(eatery_id, category, start_epoch, end_epoch, tag_analysis_algorithm_name)
                        review_ids, sentences = zip(*[(__e[0], __e[1]) for __e in __sentences])
                        
                        
                        file_path = os.path.dirname(os.path.abspath(__file__))
                        classifier_path = "{0}/Text_Processing/PrepareClassifiers/InMemoryClassifiers/".format(file_path)
                        classifier = joblib.load("{0}{1}".format(classifier_path, "svm_linear_kernel_classifier_ambience.lib"))
                        classifier_sentiment = joblib.load("{0}{1}".format(classifier_path, "svm_linear_kernel_classifier_sentiment.lib"))


                        result = list()
                        __predicted_tuples = zip(classifier.predict(sentences), classifier_sentiment.predict(sentences))


                        for __e in __predicted_tuples:
                            if __e[0] == 'ambience-null':
                                    pass
                            elif __e[1].startswith("super"):
                                result.append((__e[0], __e[1].split("-")[1]))
                                result.append((__e[0], __e[1].split("-")[1]))
                            elif __e[1] == "neutral":
                                    pass
                            else:
                                result.append((__e[0], __e[1]))

                        final_result = dict()
                        result = Counter(result)
                        for k, v in result.items():
                                final_result.update({k[0]: {"positive": 0, "negative": 0}})
                                
                        for k, v in result.items():
                                if final_result.get(k[0]):
                                        __l = final_result.get(k[0])
                                        __l.update({k[1]: v})
                                        final_result.update({k[0]: __l})
                                else:    
                                        final_result.update({k[0]: {k[1]: v}})


                        result = list()
                        for k, v in final_result.items():
                                result.append({"name": k, "positive": v.get("positive"), "negative": v.get("negative")})
                        return {"success": True,
				"error": False,
                                 "result": result,
                                }
    
                """
                celery_chain = (ReviewIdToSentTokenize.s(eatery_id, category, start_epoch, end_epoch, tag_analysis_algorithm_name, 
                    sentiment_analysis_algorithm_name, word_tokenization_algorithm_name, pos_tagging_algorithm_name, 
                    noun_phrases_algorithm_name)| NoNounPhrasesReviews.s( category, word_tokenization_algorithm_name, 
                        pos_tagging_algorithm_name, noun_phrases_algorithm_name)| MappingList.s(ner_algorithm_name, 
                            word_tokenization_algorithm_name, pos_tagging_algorithm_name, noun_phrases_algorithm_name, 
                            tag_analysis_algorithm_name,  sentiment_analysis_algorithm_name, SentTokenizeToNP.s()))()



                while celery_chain.status != "SUCCESS":
                        pass    

		try:
			for __id in celery_chain.children[0]:    
                        	while __id.status != "SUCCESS":
                                	pass
                except IndexError as e:
			pass

		
                store_in_eatery_worker = StoreInEatery.apply_async(args=[eatery_id, category, start_epoch, end_epoch, 
                    word_tokenization_algorithm_name, pos_tagging_algorithm_name, noun_phrases_algorithm_name, np_clustering_algorithm_name])
                
                while store_in_eatery_worker.status != "SUCCESS":
                        pass
                
                
                clustering_result = Clustering.apply_async(args=[eatery_id, category, start_epoch, end_epoch, word_tokenization_algorithm_name,
                        pos_tagging_algorithm_name, noun_phrases_algorithm_name, np_clustering_algorithm_name, total_noun_phrases])
                
                while clustering_result.status != "SUCCESS":
                        pass
               

                result = clustering_result.get()
                for element in result:
                        print element


                __result = HeuristicClustering(result, eatery_name)


                return {"success": True,
				"error": False,
                                
                                "result": __result.result[0: total_noun_phrases],
                    }
                """
                celery_chain = (ReviewIdToSentTokenize.s(eatery_id, category, start_epoch, end_epoch, tag_analysis_algorithm, 
                    sentiment_analysis_algorithm)|  
                    MappingList.s(ner_algorithm, word_tokenization_algorithm, pos_tagging_algorithm, noun_phrases_algorithm, 
                         tag_analysis_algorithm,  sentiment_analysis_algorithm, SentTokenizeToNP.s()))()
                print celery_chain
                ##Waitng for the ReviewIdToSentTokenize task to finish
                while celery_chain.status != "SUCCESS":
                        pass    

                ##Waiting for the SentTokenizeToNP tasks to finish
                for id in celery_chain.children[0]:    
                        while id.status != "SUCCESS":
                                pass

                    
                ##Te above code gurantees that all the SentTokenizeToNP has been finished and now we can gather reult of these
                ##tasks
                for id in celery_chain.children[0]:    
                        result.append(id.get())
                
                ##Deleting all the results from the backends
                ids =  [__id.id for __id in celery_chain.children[0]]
                ids.extend([celery_chain.parent.id, celery_chain.id])
               

                result = list(itertools.chain.from_iterable(result))
                clustering_result = Clustering.apply_async(args=[result, np_clustering_algorithm, total_noun_phrases])
                
                while clustering_result.status != "SUCCESS":
                        pass

                ids.append(clustering_result.id)
                CleanResultBackEnd.apply_async(args=[ids])


                return {"success": True,
				"error": False,
				"result": clustering_result.get(),
                    }
                """ 


class UpdateClassifier(restful.Resource):
        @cors
        @timeit
        def post(self):
                """
                Update the classifier with new data into the InMemoryClassifiers folder
                """
                args = update_classifiers.parse_args()    
                return



class ChangeTagOrSentiment(restful.Resource):
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
                whether_allowed = args["whether_allowed"]

                if not whether_allowed:
                        return {"success": False,
                                "error": True,
                                "messege": "Right now, Updating Tags or sentiments are not allowed",
                                }


                tag_list = ["food", "service", "cost", "null", "ambience", "overall"]
                sentiment_list = ["positive", "super-positive", "neutral", "negative", "super-negative", "mixed"]

                print value, sentence
                if not value in (tag_list+sentiment_list):
                        return {"success": False,
                                "error": True,
                                "messege": "Error occured",
                                }

                if value in ["food", "service", "cost", "null", "ambience", "overall"]:
                        training_tag_collection.update({"sentence": sentence}, {"$set": {
                                    "review_id": "misc",
                                    "tag": value, }}, upsert=True)
                        print "tag updated"

                if value in ["positive", "super-positive", "neutral", "negative", "super-negative"]:
                        training_sentiment_collection.update({"sentence": sentence}, {"$set": {
                                    "review_id": "misc",
                                    "sentiment": value,
                            }}, upsert=True)
                        print "sentiment updated"
                return {"success": True,
                        "error": False,
                        "messege": "Updated!!!",
                        }


class RawTextParser(restful.Resource):
        @cors
        @timeit
        def post(self):
                result = list()

                word_tokenization_algorithm = "punkt_n_treebank"
                noun_phrases_algorithm = "textblob_np_conll"
                pos_tagging_algorithm = "hunpos_pos_tagger"
                
                path = "/home/kaali/Programs/Python/Canworks/Canworks/Text_Processing/PrepareClassifiers/InMemoryClassifiers/"
                args = raw_text_processing_parser.parse_args()    
                text = args["text"]
                

                text = text.replace("\n", "")
                sentiment_result = list()
                sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                tokenized_sentences = sent_tokenizer.tokenize(text)
                print tokenized_sentences

                tag_classifier = joblib.load("{0}{1}".format(path, "svm_linear_kernel_classifier_tag.lib"))
                sentiment_classifier = joblib.load("{0}{1}".format(path, "svm_linear_kernel_classifier_sentiment.lib"))
                new_sentiment_classifier = joblib.load("{0}{1}".format(path, "svm_linear_kernel_classifier_sentiment_new_dataset.lib"))
                for sentence in zip(tokenized_sentences, tag_classifier.predict(tokenized_sentences), 
                        sentiment_classifier.predict(tokenized_sentences), new_sentiment_classifier.predict(tokenized_sentences)):
                        print sentence, "\n"
                        
                        word_tokenize = WordTokenize([sentence[0]],  default_word_tokenizer= word_tokenization_algorithm)
                        word_tokenization_algorithm_result = word_tokenize.word_tokenized_list.get(word_tokenization_algorithm)
                        
                        __pos_tagger = PosTaggers(word_tokenization_algorithm_result,  default_pos_tagger=pos_tagging_algorithm)
                        pos_tagging_algorithm_result =  __pos_tagger.pos_tagged_sentences.get(pos_tagging_algorithm)

                        __noun_phrases = NounPhrases(pos_tagging_algorithm_result, default_np_extractor=noun_phrases_algorithm)
                        noun_phrases_algorithm_result =  __noun_phrases.noun_phrases.get(noun_phrases_algorithm)

                        result.extend([[noun, sentence[2]] for noun in flatten(noun_phrases_algorithm_result)])
                        sentiment_result.append([sentence[0], sentence[1], sentence[3]])

                print result
		
                edited_result = list()
                for element in result:
                        if element[1].startswith("super"):
                                edited_result.append((element[0], element[1].split("-")[1]))
                                edited_result.append((element[0], element[1].split("-")[1]))
                        else:
                                edited_result.append(tuple(element)) 
                print edited_result

                final_result = list()
                for key, value in Counter(edited_result).iteritems():
                        final_result.append({"name": key[0], "polarity": 1 if key[1] == 'positive' else 0 , "frequency": value})
                
                sorted_result = sorted(final_result, reverse=True, key=lambda x: x.get("frequency"))
                for element in sorted_result:
                    print sorted_result

                __result = HeuristicClustering(sorted_result, None)

                for element in __result.result:
                    print element

                def convert_sentences(__object):
                        return {"sentence": __object[0], 
                                "tag": __object[1],
                                "sentiment": __object[2], }


                return {"success": True,
				"error": False,
                                "result": __result.result,
                                "sentences": map(convert_sentences, sentiment_result),
                                }

class TestWhole(restful.Resource):
	@cors
	@timeit
        def post(self):
                def solve_key_error_threading():
                        if 'threading' in sys.modules:
                                del sys.modules['threading']
                        import gevent
                        import gevent.socket
                        import gevent.monkey
                        gevent.monkey.patch_all()

                #solve_key_error_threading()
                args = test_whole_parser.parse_args()    
                print args
                """


                review_text = [to_unicode_or_bust(post.get("review_text")) for post in reviews.find({"eatery_id": "4571"})]
		
                review_text = " .".join(review_text)
                sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                __word_tokenize = WordTokenize(sent_tokenizer.tokenize(review_text)) #using punkt_n_treebank_tokenizer
                word_tokenized_list =  __word_tokenize.word_tokenized_list
               

                __pos_tagger = PosTaggers(word_tokenized_list.get("punkt_n_treebank")[82:100]) #using default standford pos tagger
                __pos_tagged_sentences =  __pos_tagger.pos_tagged_sentences
                
                print __pos_tagged_sentences
                
                __noun_phrases = NounPhrases(__pos_tagged_sentences.get("stan_pos_tagger"))
                print __noun_phrases.noun_phrases
                """
                return   {"success": True,
                            "error": False,
                            "result": result.get(),
                            }

api.add_resource(EateriesList, '/eateries_list')
api.add_resource(LimitedEateriesList, '/limited_eateries_list')
api.add_resource(GetWordCloud, '/get_word_cloud')
api.add_resource(FBLogin, '/fb_login')
api.add_resource(PostComment, '/post_comment')
api.add_resource(SuggestName, '/name_suggestion')
api.add_resource(PostPicture, '/post_pic')
api.add_resource(GetPics, '/get_pics')
api.add_resource(GetStartDateForRestaurant, '/get_start_date_for_restaurant') 
api.add_resource(RawTextParser, '/raw_text_processing') 
api.add_resource(ChangeTagOrSentiment, '/change_tag_or_sentiment')

api.add_resource(TestWhole, '/test') 

if __name__ == '__main__':
        app.run(port=8000, debug=True)
