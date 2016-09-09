#!/usr/bin/env python
"""
Author: kaali
Date: 16 May, 2015
Purpose: Final celery code to be run with tornado
"""
import pymongo
import os
import sys
import warnings
import itertools

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_path)
from GlobalConfigs import reviews, eateries, reviews_results_collection, \
        eateries_results_collection, bcolors

#Algortihms of the form .lib
from GlobalAlgorithmNames import TAG_CLASSIFIER, SENTI_CLASSIFIER, FOOD_SB_TAG_CLASSIFIER, \
        COST_SB_TAG_CLASSIFIER, SERV_SB_TAG_CLASSIFIER, AMBI_SB_TAG_CLASSIFIER

#Name of the algortihms      
from GlobalAlgorithmNames import NOUN_PHSE_ALGORITHM_NAME, TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME,\
        FOOD_SB_CLSSFY_ALG_NME, SERV_SB_CLSSFY_ALG_NME, AMBI_SB_CLSSFY_ALG_NME, COST_SB_CLSSFY_ALG_NME

##Actual libraries loaded by joblib.load
from GlobalAlgorithmNames import TAG_CLASSIFIER_LIB, SENTI_CLASSIFIER_LIB, FOOD_SB_TAG_CLASSIFIER_LIB,\
        COST_SB_TAG_CLASSIFIER_LIB , SERV_SB_TAG_CLASSIFIER_LIB, AMBI_SB_TAG_CLASSIFIER_LIB


from GlobalAlgorithmNames import FOOD_SUB_TAGS, COST_SUB_TAGS, SERV_SUB_TAGS, AMBI_SERV_TAGS



class MongoScriptsReviews(object):
        
        @staticmethod
        def return_reviews(eatery_id, start_epoch=None, end_epoch=None):
                if start_epoch and end_epoch:
                        review_list = [post.get("review_id") for post in reviews.find({"eatery_id" :eatery_id, \
                            "converted_epoch": {"$gt":  start_epoch, "$lt" : end_epoch}})]
                else:
                        review_list = [post.get("review_id") for post in reviews.find({"eatery_id" :eatery_id})]
                    
                return review_list

        @staticmethod
        def return_all_reviews(eatery_id):
                review_list = [post.get("review_id") for post in reviews.find({"eatery_id" :eatery_id})]
                return review_list

        @staticmethod
        def return_all_reviews_with_text(eatery_id):
                review_list = [(post.get("review_id"), post.get("review_text")) for post \
                        in reviews.find({"eatery_id" :eatery_id})]
                return review_list


        @staticmethod
        def reviews_with_text(reviews_ids):
                review_list = list()
                for review_id in reviews_ids:
                        review_text = reviews.find_one({"review_id": review_id}).get("review_text")
                        review_list.append((review_id, review_text))
                return review_list



class MongoScriptsEateries(object):
        def __init__(self, eatery_id):
                    self.eatery_id = eatery_id

        def check_algorithms(self):
                
                if eateries_results_collection.find_one({"eatery_id": self.eatery_id, 
                    "tag_anlysis_algorithm_name": TAG_CLASSIFY_ALG_NME,
                    "sentiment_analysis_algorithm_name": SENTI_CLSSFY_ALG_NME,
                    "food_classification_algorithm_name": FOOD_SB_CLSSFY_ALG_NME,
                    "cost_classification_algorithm_name": COST_SB_CLSSFY_ALG_NME,
                    "service_classification_algorithm_name": SERV_SB_CLSSFY_ALG_NME,
                    "ambience_classification_algorithm_name": AMBI_SB_CLSSFY_ALG_NME,
                    "noun_phrases_algorithm_name": NOUN_PHSE_ALGORITHM_NAME}):
                        return True
        
                warnings.warn("{0}The set of algorithms for eatery --<<{1}>>-- has not been found\
                        {2}".format(bcolors.FAIL, self.eatery_id, bcolors.RESET))
                return False


        def empty_processed_reviews_list(self):
                if eateries_results_collection.find_one({"eatery_id": self.eatery_id}):
                        eateries_results_collection.update({"eatery_id": self.eatery_id}, \
                                {"$unset": {"processed_reviews": None}})
                return

        
        def empty_old_considered_ids(self):
                if eateries_results_collection.find_one({"eatery_id": self.eatery_id}):
                        eateries_results_collection.update({"eatery_id": self.eatery_id}, \
                                {"$unset": {"old_considered_ids": None}})
                return
        
        def empty_noun_phrases(self):
                if eateries_results_collection.find_one({"eatery_id": self.eatery_id}):
                        eateries_results_collection.update({"eatery_id": self.eatery_id}, \
                                {"$unset": {"noun_phrases": None}})
                return



        def set_new_algorithms(self):
                
                eateries_results_collection.update({"eatery_id": self.eatery_id}, {"$set": {
                    "tag_anlysis_algorithm_name": TAG_CLASSIFY_ALG_NME,
                    "sentiment_analysis_algorithm_name": SENTI_CLSSFY_ALG_NME,
                    "food_classification_algorithm_name": FOOD_SB_CLSSFY_ALG_NME,
                    "cost_classification_algorithm_name": COST_SB_CLSSFY_ALG_NME,
                    "service_classification_algorithm_name": SERV_SB_CLSSFY_ALG_NME,
                    "ambience_classification_algorithm_name": AMBI_SB_CLSSFY_ALG_NME,
                    "noun_phrases_algorithm_name": NOUN_PHSE_ALGORITHM_NAME}}, upsert=True)

                warnings.warn("{0}Setting new algorithms for eatery --<<{1}>>-- has not been found\
                        {2}".format(bcolors.OKBLUE, self.eatery_id, bcolors.RESET))
                return


        def get_proccessed_reviews(self):
                result = eateries_results_collection.find_one({"eatery_id": self.eatery_id}, {"_id": False, 
                                    "processed_reviews": True})
    
                return result.get("processed_reviews")


class MongoScripts:
        def __init__(self):
                pass


        @staticmethod
        def get_review_result(review_id):
                result = dict()

                #Everything has to be changed because now sentiments has to be changed
                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME)\
                            : {"$exists": True}}))):
                        result.update({"rerun_all_algorithms": True})
                        return result

                #Ony the food category has to be changed
                #only the fields corresponding to 
                #tag_analysis_algorithm_name.sentiment_analysis_algorithm_name.food_classifier_algorithm_name
                #needs to be changed, rest remains the same
                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}.food.{2}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME\
                            , FOOD_SB_CLSSFY_ALG_NME) : {"$exists": True}}))):
                        result.update({"rerun_food_sub_tag_classification": True})
                


                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}.food.{2}.{3}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME\
                            ,FOOD_SB_CLSSFY_ALG_NME, NOUN_PHSE_ALGORITHM_NAME) : {"$exists": True}}))):
                        result.update({"rerun_noun_phrases": True})


                
                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}.cost.{2}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME\
                            , COST_SB_CLSSFY_ALG_NME) : {"$exists": True}}))):
                        result.update({"rerun_cost_sub_tag_classification": True})


                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}.service.{2}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME\
                            , SERV_SB_CLSSFY_ALG_NME) : {"$exists": True}}))):
                        result.update({"rerun_service_sub_tag_classification": True})


                if not bool(list(reviews_results_collection.find({'review_id': review_id, 
                    "classification.{0}.{1}.ambience.{2}".format(TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME\
                            , AMBI_SB_CLSSFY_ALG_NME) : {"$exists": True}}))):
                        result.update({"rerun_ambience_sub_tag_classification": True})

                return result

    
        @staticmethod
        def update_processed_reviews_list(eatery_id, review_id):
                eateries_results_collection.update({"eatery_id": eatery_id}, {"$push": {
                                "processed_reviews": review_id}}, upsert=False)

                warnings.warn("{0} Updating processed_review list of eatery --<<{1}>>-- with review_id\
                        {2}{3}".format(bcolors.OKBLUE, eatery_id, review_id, bcolors.RESET))
                return

        @staticmethod
        def get_tag_sentences(review_id, tag):
                return reviews_results_collection.find_one({"review_id": review_id}).get(tag)

        
        @staticmethod
        def update_food_sub_tag_sentences(review_id, food_result):
                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"classification.{0}.{1}.food.{2}.{3}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, FOOD_SB_CLSSFY_ALG_NME, NOUN_PHSE_ALGORITHM_NAME):
                            food_result,}}, upsert=False, multi=False)
        
                return

        @staticmethod
        def update_cost_sub_tag_sentences(review_id, cost_result):
                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"classification.{0}.{1}.cost.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, COST_SB_CLSSFY_ALG_NME): cost_result,
                    }}, upsert=False, multi=False)
                return
                    
        @staticmethod
        def update_service_sub_tag_sentences(review_id, service_result):
                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"classification.{0}.{1}.service.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, SERV_SB_CLSSFY_ALG_NME): service_result,
                    }}, upsert=False, multi=False)
                return

        @staticmethod
        def update_ambience_sub_tag_sentences(review_id, ambience_result):
                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"classification.{0}.{1}.ambience.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, AMBI_SB_CLSSFY_ALG_NME): ambience_result,
                    }}, upsert=False, multi=False)
                return

        @staticmethod
        def update_noun_phrases(review_id, noun_phrases_result):
                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"classification.{0}.{1}.food.{2}.{3}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, FOOD_SB_CLSSFY_ALG_NME, NOUN_PHSE_ALGORITHM_NAME):
                            noun_phrases_result,}}, upsert=False, multi=False)

                return

        @staticmethod
        def update_review_result_collection(**kwargs):
                review_id = kwargs["review_id"]
                eatery_id = kwargs["eatery_id"]
                food_sentences = kwargs["food"]
                cost_sentences = kwargs["cost"]
                service_sentences = kwargs["service"]
                ambience_sentences = kwargs["ambience"]
                null_sentences = kwargs["null"]
                overall_sentences = kwargs["overall"]
                food_result = kwargs["food_result"]
                service_result = kwargs["service_result"]
                cost_result = kwargs["cost_result"]
                ambience_result = kwargs["ambience_result"]

                review = reviews.find_one({"review_id": review_id}, {"_id": False, "review_text": True, \
                        "converted_epoch": True, "review_time": True})    
               

                print reviews_results_collection.update({"review_id": review_id}, {"$set": 
                    {"food": food_sentences, "cost": cost_sentences, "ambience": ambience_sentences, \
                            "service": service_sentences, "null": null_sentences, "overall": overall_sentences, 
                            "review_text": review.get("review_text"), "review_time": review.get("review_time"), 
                            "eatery_id": eatery_id, 
                    "classification.{0}.{1}.food.{2}.{3}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, FOOD_SB_CLSSFY_ALG_NME, NOUN_PHSE_ALGORITHM_NAME):
                            food_result,

                    "classification.{0}.{1}.cost.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, COST_SB_CLSSFY_ALG_NME): cost_result,

                    "classification.{0}.{1}.ambience.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, AMBI_SB_CLSSFY_ALG_NME): ambience_result,
                    
                    "classification.{0}.{1}.service.{2}".format(TAG_CLASSIFY_ALG_NME, 
                        SENTI_CLSSFY_ALG_NME, SERV_SB_CLSSFY_ALG_NME): service_result,
                    
                    }}, upsert=True, multi=False)

                return 
                

class MongoScriptsDoClusters(object):
        def __init__(self, eatery_id):
                self.eatery_id = eatery_id
                

        def processed_clusters(self):
                """
                This returns all the noun phrases that already have been processed for
                teh eatery id 
                """
                return eateries_results_collection.find_one({"eatery_id": self.eatery_id}, 
                            {"_id": False, noun_phrases: True}).get("noun_phrases")

        def old_considered_ids(self):
                """
                Returns review_ids whose noun_phrases has already been taken into account, 
                which means Clusteirng algorithms has already been done on the noun phrases 
                of these review ids and is stored under noun_phrases key of eatery
                nd these review ids has been stored under old_considered_ids
                """
                return eateries_results_collection.find_one({"eatery_id": self.eatery_id}, {"_id": False, 
                            "old_considered_ids": True}).get("old_considered_ids")


        def processed_reviews(self):
                return eateries_results_collection.find_one({"eatery_id": self.eatery_id}, {"_id": False, 
                            "processed_reviews": True}).get("processed_reviews")

        def fetch_reviews(self, category, review_list=None):
                if not review_list:
                        review_list = eateries_results_collection.find_one({"eatery_id": \
                                self.eatery_id}).get("processed_reviews")


                if category == "food":
                        food = [reviews_results_collection.find_one({"review_id": review_id})\
                            ["classification"][TAG_CLASSIFY_ALG_NME][SENTI_CLSSFY_ALG_NME]\
                            ["food"][FOOD_SB_CLSSFY_ALG_NME][NOUN_PHSE_ALGORITHM_NAME] for review_id in  review_list]
                        new_food = list(itertools.chain(*food))
                        
                        food_reviews_text = [reviews_results_collection.find_one({"review_id": review_id}).get("review_text")
                                for review_id in review_list]

                        ##Checks if the food sub classification tab has beenchanged or not
                        if list(set([e[3] for e in food])) != FOOD_SUB_TAGS:
                                raise StandardError("Food sub classification has been changed,\
                                            Do something about it fuckerrrr, One way to solve this problem\
                                            is to flush whole database with food category, also food noun\
                                            phrases stored in the eatery")


                        return new_food               
                        
                        
       
                    
                result = [post["classification"][TAG_CLASSIFY_ALG_NME][SENTI_CLSSFY_ALG_NME][category]\
                        [eval("{0}_SB_CLSSFY_ALG_NME".format(category.upper()[0:4]))] for post in \
                        reviews_results_collection.find()]
                
                result = list(itertools.chain(*result))
                if list(set([e[3] for e in result])) != eval("{0}_SUB_TAGS".format(category.upper()[0:4])):
                        raise StandardError("{0} sub classification has been changed,\
                                            Do something about it fuckerrrr, One way to solve this problem\
                                            is to flush whole database with {0} category".format(category))

               
                #[[u'super-positive', u'ambience-overall'], [u'super-positive', u'ambience-overall'], 
                #[u'neutral', u'ambience-overall']]
                return [[sentiment, sub_tag] for (sent, tag, sentiment, sub_tag) in result]

        
        def fetch_nps_frm_eatery(self, category):
                return eateries_results_collection.find_one({"eatery_id": self.eatery_id}).get(category)

        def update_considered_ids(self, review_list=None):
                """
                Updating considered ids, Adding review ids to eateries considered_ids list, Thre reviews ids
                for which the noun phrases of all categories has been added to respective lists in eateries
                
                if reviews_list is not present updated old_considered_ids with all the ids present 
                in processed_reviews list of the eatery
                """
                if not review_list:
                        reviews_list = eateries_results_collection.find_one({"eatery_id":\
                                    self.eatery_id}).get("processed_reviews")
                
                try:
                        eateries_results_collection.update({"eatery_id": self.eatery_id}, {"$push": \
                                {"old_considered_ids": {"$each": review_list }}}, upsert=False)

                except Exception as e:
                        raise StandardError(e)
                return

        def update_nps(self, category, category_nps):
                """
                Update new noun phrases to the eatery category list
                """
                try:
                    eateries_results_collection.update({"eatery_id": self.eatery_id}, {"$set": {category:
                    category_nps}}, upsert=False)
                except Exception as e:
                    raise StandardError(e)
                return 


        def fetch_reviews_for_review_list(self, review_ids_list, category):
                """
                Fetch reviews from review_list 

                """
                return 






