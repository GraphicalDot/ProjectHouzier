#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: kaali
Dated: 15 April, 2015
Purpose:
    This file has been written to list all the sub routines that might be helpful in generating result for 
    get_word_cloud api

"""
import time
import os
from sys import path
import itertools
import warnings
from sklearn.externals import joblib
from collections import Counter
from mongo_scripts import MongoScripts, MongoScriptsEateries, MongoScriptsReviews

parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path.append(parent_dir_path)

from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
from Text_Processing.NounPhrases.noun_phrases import NounPhrases
from Text_Processing.Word_Tokenization.word_tokenizer import WordTokenize
from Text_Processing.PosTaggers.pos_tagging import PosTaggers
from Text_Processing.MainAlgorithms.paths import path_parent_dir, path_in_memory_classifiers
from Text_Processing.NER.ner import NERs
from Text_Processing.colored_print import bcolors
from Text_Processing.MainAlgorithms.Algorithms_Helpers import get_all_algorithms_result
from Text_Processing.MainAlgorithms.In_Memory_Main_Classification import timeit, cd
from encoding_helpers import SolveEncoding
from heuristic_clustering_without_sentences import HeuristicClustering 



from GlobalAlgorithmNames import TAGS
#Algortihms of the form .lib
from GlobalAlgorithmNames import TAG_CLASSIFIER, SENTI_CLASSIFIER, FOOD_SB_TAG_CLASSIFIER,\
        COST_SB_TAG_CLASSIFIER, SERV_SB_TAG_CLASSIFIER, AMBI_SB_TAG_CLASSIFIER 
      
#Name of the algortihms      
from GlobalAlgorithmNames import NOUN_PHSE_ALGORITHM_NAME, TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME,\
        FOOD_SB_CLSSFY_ALG_NME, SERV_SB_CLSSFY_ALG_NME, AMBI_SB_CLSSFY_ALG_NME, COST_SB_CLSSFY_ALG_NME

##Actual libraries loaded by joblib.load
from GlobalAlgorithmNames import TAG_CLASSIFIER_LIB, SENTI_CLASSIFIER_LIB, FOOD_SB_TAG_CLASSIFIER_LIB,\
        COST_SB_TAG_CLASSIFIER_LIB , SERV_SB_TAG_CLASSIFIER_LIB, AMBI_SB_TAG_CLASSIFIER_LIB 

class EachEatery:
        def __init__(self, eatery_id):
                self.eatery_id = eatery_id
                self.mongo_eatery_instance = MongoScriptsEateries(self.eatery_id)
        
        def return_non_processed_reviews(self, start_epoch=None, end_epoch=None):
                
                ##If there is a change in algortihms or a new eatery to be processed
                #we run processing all the reviews independent of the start_epoch and
                #end_epoch, which means whenever there is change all the revviews will
                #be present in processed_reviews list of eatery unless celery fails to
                #process all, in case of some internal error
                if not self.mongo_eatery_instance.check_algorithms():
                        self.mongo_eatery_instance.empty_processed_reviews_list()
                        self.mongo_eatery_instance.set_new_algorithms()
                        self.mongo_eatery_instance.empty_noun_phrases()
                        self.mongo_eatery_instance.empty_old_considered_ids()
            
                        return MongoScriptsReviews.return_all_reviews_with_text(self.eatery_id)

                else:
                        all_reviews = MongoScriptsReviews.return_all_reviews(self.eatery_id) 
                        all_processed_reviews = self.mongo_eatery_instance.get_proccessed_reviews()

                        ##This if True means that the database has some new reviews added to it, 
                        ##which needs processing, so MongoScriptsReviews.reviews_with_text 
                        #returns (review_id, review_text) for every review_id
                        if all_reviews != all_processed_reviews:
                                warnings.warn("{0} we encountered new reviews in the database {1}".format(\
                                        bcolors.FAIL, bcolors.RESET))
                                
                                try:
                                        reviews_ids = list(set.symmetric_difference(set(all_reviews), set(all_processed_reviews)))
                                        return MongoScriptsReviews.reviews_with_text(reviews_ids)
                                #This means that all_processed_reviews is empty, which means all the reviews
                                ##needs processing 
                                except TypeError as e:
                                        warnings.warn("{0} It seems none of the review has been processed yet {1}".format(\
                                        bcolors.FAIL, bcolors.RESET))
                                        return MongoScriptsReviews.reviews_with_text(all_reviews)


class PerReview:
        sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
        def __init__(self, review_id, review_text, eatery_id):
                
                self.review_id, self.review_text, self.eatery_id = review_id, review_text, eatery_id

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
        
        def get_args(self):
                print self.__dict__
        

        @print_execution
        def run(self):
                """
                It returns the result
                """

                result = self.__get_review_result()
                print "this is the result %s"%result
                if not bool(result):
                        print "{0}Result for the review_id --<<{1}>>-- has alredy been found{2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        return 


                if result.get("rerun_food_sub_tag_classification"):

                        print "{0}Doing FOOD sub classification again for review_id --<<{1}>>-- {2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        self.food = MongoScripts.get_tag_sentences(self.review_id, "food")
                        self.__food_sub_tag_classification()
                        self.__extract_noun_phrases() #makes self.noun_phrases
                        MongoScripts.update_food_sub_tag_sentences(self.review_id, self.all_food_with_nps)
               

                if result.get("rerun_cost_sub_tag_classification"):
                        print "{0}Doing COST sub classification again for review_id --<<{1}>>-- {2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        self.cost = MongoScripts.get_tag_sentences(self.review_id, "cost")
                        
                        self.__cost_sub_tag_classification()
                        MongoScripts.update_cost_sub_tag_sentences(self.review_id, self.all_cost, 
                                TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME, 
                                COST_SB_CLSSFY_ALG_NME)


                if result.get("rerun_service_sub_tag_classification"):
                        print "{0}Doing SERVICE sub classification again for review_id --<<{1}>>-- {2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        self.service = MongoScripts.get_tag_sentences(self.review_id, "service")
                        self.__service_sub_tag_classification()
                        MongoScripts.update_service_sub_tag_sentences(self.review_id, self.all_service, 
                                TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME, SERV_SB_CLSSFY_ALG_NME)

                if result.get("rerun_ambience_sub_tag_classification"):
                        print "{0}Doing AMBIENCE sub classification again for review_id --<<{1}>>-- {2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        self.ambience = MongoScripts.get_tag_sentences(self.review_id, "ambience")
                        self.__ambience_sub_tag_classification()
                        MongoScripts.update_ambience_sub_tag_sentences(self.review_id, self.all_ambience, 
                                TAG_CLASSIFY_ALG_NME, SENTI_CLSSFY_ALG_NME, AMBI_SB_CLSSFY_ALG_NME)
                        

                if result.get("rerun_noun_phrases"):
                        print "{0}Doing noun phrases again for review_id --<<{1}>>-- {2}".format(bcolors.OKBLUE, \
                                self.review_id, bcolors.RESET)
                        
                        self.food = MongoScripts.get_tag_sentences(self.review_id, "food")
                        self.__food_sub_tag_classification()
                        self.__extract_noun_phrases() #makes self.noun_phrases
                        MongoScripts.update_noun_phrases(review_id, self.all_food_with_nps, TAG_CLASSIFY_ALG_NME,\
                                SENTI_CLSSFY_ALG_NME, FOOD_SB_CLSSFY_ALG_NME, NOUN_PHRASES_ALGORITHM_NAME)
        


                if result.get("rerun_all_algorithms"):
                        print "{0} No results found for review id --<<{1}>>--{2}".format(bcolors.FAIL, \
                                self.review_id, bcolors.RESET)
                        self.__sent_tokenize_review() #Tokenize reviews, makes self.reviews_ids, self.sentences
                        self.__predict_tags()          #Predict tags, makes self.predict_tags
                        self.__predict_sentiment() #makes self.predicted_sentiment

                        self.all_sent_tag_sentiment = zip(self.sentences, self.tags, self.sentiments)
                
                        self.__filter_on_category() #generates self.food, self.cost, self.ambience, self.service
                

                        self.__food_sub_tag_classification()
                        self.__service_sub_tag_classification()
                        self.__cost_sub_tag_classification()
                        self.__ambience_sub_tag_classification()

                        self.__extract_noun_phrases() #makes self.noun_phrases
                        self.__update_review_result()
                
                MongoScripts.update_processed_reviews_list(self.eatery_id, self.review_id)
                return 

        @print_execution
        def __food_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.food_sub_tags = FOOD_SB_TAG_CLASSIFIER_LIB.predict([__e[0] for __e in self.food])
                self.all_food = [(sent, tag, sentiment, sub_tag) for ((sent, tag, sentiment), sub_tag)\
                        in zip(self.food, self.food_sub_tags)]

                return 
       

        @print_execution
        def __service_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.service_sub_tags = SERV_SB_TAG_CLASSIFIER_LIB.predict([__e[0] for __e in self.service])
                self.all_service = [(sent, tag, sentiment, sub_tag) for ((sent, tag, sentiment), sub_tag) \
                        in zip(self.service, self.service_sub_tags)]
                
                return 

        @print_execution
        def __cost_sub_tag_classification(self):
                """
                This deals with the sub classification of cost sub tags
                
                self.all_cost = [(sent, "cost", sentiment, "cost-overall",), .....]
                """

                self.cost_sub_tags = COST_SB_TAG_CLASSIFIER_LIB.predict([__e[0] for __e in self.cost])
                self.all_cost = [(sent, tag, sentiment, sub_tag) for ((sent, tag, sentiment), sub_tag) \
                        in zip(self.cost, self.cost_sub_tags)]
                
                return 

        @print_execution
        def __ambience_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.ambience_sub_tags = AMBI_SB_TAG_CLASSIFIER_LIB.predict([__e[0] for __e in self.ambience])
                self.all_ambience = [(sent, tag, sentiment, sub_tag) for ((sent, tag, sentiment), sub_tag) \
                        in zip(self.ambience, self.ambience_sub_tags)]
                
                return




        @print_execution
        def __sent_tokenize_review(self):
                """
                Tokenize self.reviews tuples of the form (review_id, review) to sentences of the form (review_id, sentence)
                and generates two lists self.review_ids and self.sentences
                """
                self.sentences = self.sent_tokenizer.tokenize(self.review_text)
                return
                       

        @print_execution
        def __predict_tags(self):
                """
                Predict tags of the sentence which were being generated by self.sent_tokenize_reviews
                """
                self.tags = TAG_CLASSIFIER_LIB.predict(self.sentences)
                return

        @print_execution
        def __predict_sentiment(self):
                """
                Predict sentiment of self.c_sentences which were made by filtering self.sentences accoring to 
                the specified category
                """
                self.sentiments = SENTI_CLASSIFIER_LIB.predict(self.sentences)
                return 
        


        @print_execution
        def __filter_on_category(self):
                 __filter = lambda tag, __list: [(sent, __tag, sentiment) for (sent, __tag, sentiment) in \
                                                                                    __list if __tag== tag ]


                 self.food, self.cost, self.ambience, self.service, self.null, self.overall = \
                         __filter("food", self.all_sent_tag_sentiment),  __filter("cost", self.all_sent_tag_sentiment),\
                         __filter("ambience", self.all_sent_tag_sentiment), __filter("service", self.all_sent_tag_sentiment),\
                         __filter("null", self.all_sent_tag_sentiment),  __filter("overall", self.all_sent_tag_sentiment)


        @print_execution
        def __extract_noun_phrases(self):
                """
                Extarct Noun phrases for the self.c_sentences for each sentence and outputs a list 
                self.sent_sentiment_nps which is of the form 
                [('the only good part was the coke , thankfully it was outsourced ', 
                                            u'positive', [u'good part']), ...]
                """
                __nouns = NounPhrases([e[0] for e in self.all_food], default_np_extractor=NOUN_PHSE_ALGORITHM_NAME)

                self.all_food_with_nps = [(sent, tag, sentiment, sub_tag, nps) for ((sent, tag, sentiment, sub_tag,), nps) in 
                        zip(self.all_food, __nouns.noun_phrases[NOUN_PHSE_ALGORITHM_NAME])]

                return self.all_food_with_nps


        @print_execution
        def __get_review_result(self):
                result = MongoScripts.get_review_result(review_id = self.review_id) 
                return result 

        
        @print_execution
        def __update_review_result(self):
                MongoScripts.update_review_result_collection(
                        review_id = self.review_id, 
                        eatery_id = self.eatery_id, 
                        food = self.food,
                        cost = self.cost,
                        ambience = self.ambience,
                        null = self.null,
                        overall = self.overall,
                        service = self.service, 
                        food_result= self.all_food_with_nps, 
                        service_result = self.all_service, 
                        cost_result = self.all_cost, 
                        ambience_result = self.all_ambience, ) 
                return 


class DoClusters(object):
        """
        Does heuristic clustering for the all reviews
        """
        def __init__(self, eatery_id, category, with_sentences=False):
                self.eatery_id = eatery_id
                self.mongo_instance = MongoScriptsDoClusters(self.eatery_id)
                self.category = category


        def run(self):
                """
                Two cases:
                    Case 1:Wither first time the clustering is being run, The old_considered_ids list
                    is empty
                        case food:

                        case ["ambience", "cost", "service"]:
                            instance.fetch_reviews(__category) fetch a list of this kind
                            let say for ambience [["positive", "ambience-null"], ["negative", "decor"], 
                            ["positive", "decor"]...]
                            
                            DoClusters.make_cluster returns a a result of the form 
                                [{"name": "ambience-null", "positive": 1, "negative": 2}, ...]

                            After completing it for all categories updates the old_considered_ids of 
                            the eatery with the reviews

                    Case 2: None of processed_reviews and old_considered_ids are empty
                        calculates the intersection of processed_reviews and old_considered_ids
                        which implies that some review_ids in old_considered_ids are not being 
                        considered for total noun phrases
                        
                        Then fecthes the review_ids of reviews who are not being considered for total
                        noun phrases 

                        instance.fetch_reviews(__categoryi, review_ids) fetch a list of this kind
                            let say for ambience [["positive", "ambience-null"], ["negative", "decor"], 
                            ["positive", "decor"]...]

                """

                old_considered_ids = self.mongo_instance.old_considered_ids()
                if not old_considered_ids:
                        #That clustering is running for the first time
                        warnings.warn("No clustering of noun phrases has been done yet")
                        instance = MongoScriptsDoClusters(self.eatery_id)
                        
                        __nps_food = instance.fetch_reviews("food")
                        sub_tag_dict = DoClusters.unmingle_food_sub(__nps_food)
                        dishes_result = DoClusters.clustering(sub_tag_dict.get("dishes"))
                        
                        #Result corresponding to the place-food tag
                        place_food_result = DoClusters.clustering(sub_tag_dict.get("place-food"))
                        
                        #REsult corresponding to the sub-food tag
                        sub_food_result = DoClusters.clustering(sub_tag_dict.get("sub-food"))

                        #REsult corresponding to the menu-food tag
                        menu_result = DoClusters.aggregation(sub_tag_dict.get("menu-food"))
                        
                        #REsult corresponding to the overall-food tag
                        overall_result = DoClusters.aggregation(sub_tag_dict.get("overall-food"))



                        for __category in ["ambience", "service", "cost"]:
                                __nps = instance.fetch_reviews(__category)
                                __whle_nps = DoClusters.make_cluster(__nps)
                                self.mongo_instance.update_nps(__category, __whle_nps)
                        
                        self.mongo_instance.update_considered_ids()
                else:
                        
                        pocessed_reviews = self.mongo_instance.processed_reviews()
                        reviews_ids = list(set.symmetric_difference(set(old_considered_ids), \
                                set(processed_reviews)))

                        __nps_food = instance.fetch_reviews("food", reviews_ids)
                                        

                        for __category in ["ambience", "service", "cost"]:
                                __nps = instance.fetch_reviews(__category, reviews_ids)
                                __nps_new_result = DoClusters.make_cluster(__nps)

                                __nps_old_result = self.mongo_instance.fetch_nps_frm_eatery(__category)
                                __whle_nps = DoClusters.adding_new_old_nps(__nps_old_result, __nps_new_result)

                                self.mongo_instance.update_nps(__category, __whle_nps)
                        
                        self.mongo_instance.update_considered_ids(review_list=reviews_ids)






        @staticmethod
        def clustering(__dishes_list):
                """
                Args __dishes_list:
                    [[u'positive',[u'paneer chilli pepper starter']], [u'positive', []],
                    [u'positive', [u'friday night']], [u'positive', []],
                    [u'super-positive', [u'garlic flavours', u'penne alfredo pasta']]],
                """
                
                ##Removing (sentiment, nps) with empyt noun phrases
                __dishes_list = [(sentiment, nps) for (sentiment, sent, nps) in __dishes_list if nps]
                __sentences = [sent for (sentiment, sent, nps) in __dishes_list if nps]

                ##Do heuritstic clustering


        @staticmethod
        def aggregation(__list):
                """
                __list can be either menu-food, overall-food,
                as these two lists doesnt require clustering but only aggreation of sentiment analysis
                Args:
                    [[u'negative', []], [u'negative', [u'i haven']], [u'negative', []], [u'neutral', []],
                    [u'negative', []], [u'neutral', []]]

                Result:
                    {u'negative': 4, u'neutral': 2}
                """
                sentiment_dict = list()
                for sentiment, frequency in Counter([sentiment for (sentiment, nps) in __list]).iteritems():
                        sentiment_dict.update({sentiment, frequency})
                return sentiment_dict

   


        @staticmethod
        def unmingle_food_sub(__list):
                """
                __list = [u'the panner chilly was was a must try for the vegetarians from the menu .', 
                u'food', u'positive', u'menu-food',[]],
                [u'and the Penne Alfredo pasta served hot and good with lots of garlic flavours which 
                we absolute love .', u'food',cu'super-positive',u'dishes', [u'garlic flavours', 
                u'penne alfredo pasta']],

                result:
                    {u'dishes': [[u'positive', [u'paneer chilli pepper starter']],
                                [u'positive', []],
                                [u'positive', [u'friday night']],
                                [u'positive', []],
                                [u'positive', []],
                                [u'super-positive', [u'garlic flavours', u'penne alfredo pasta']]],
                    u'menu-food': [[u'positive', []]],
                    u'null-food': [[u'negative', []],
                                [u'super-positive', []],
                                [u'super-positive', []],
                                [u'negative', []],
                                }
                """
                __sub_tag_dict = dict()
                for (sent, tag, sentiment, sub_tag, nps)  in __list:
                        if not __sub_tag_dict.has_key(sub_tag):
                                __sub_tag_dict.update({sub_tag: [[sentiment, sent, nps]]})
                        
                        else:
                            __old = __sub_tag_dict.get(sub_tag)
                            __old.append([sentiment, sent, nps])
                            __sub_tag_dict.update({sub_tag: __old})

                return __sub_tag_dict

        @staticmethod
        def make_cluster(__nps):
                """
                args:
                    __nps: [[sentiment, sent, sub_tag], [sentiment, sent, sub_tag] , ...]
                return:
                        [{'name': u'ambience-null', u'positive': 1},
                        {'name': u'ambience-overall', u'neutral': 1, u'super-positive': 2}]
                    
                """
                nps_dict = DoClusters.make_sentences_dict(__nps)
                result = [DoClusters.flattening_dict(key, value) for key, value in nps_dict.iteritems()]
                return result

        @staticmethod
        def adding_new_old_nps(__new, __old):
                """
                __new: [{'name': u'decor', u'super-positive': 1, "negative": 3}, {'name': u'ambience-null', u'positive': 1}, 
                __old: [{'name': u'music', u'super-positive': 1}, {'name': u'ambience-null', u'neutral': 1, "super-negative": 10}, 
                {'name': u'ambience-overall', u'neutral': 1, u'super-positive': 2},
                """
                aggregated_list = list()
                def make_dict(__list):
                        new_dict = dict()
                        for __dict in __list:
                                name = __dict.get("name")
                                __dict.pop("name")
                                new_dict.update({name: __dict})
                        return new_dict

                __new_dict = make_dict(__new)
                __old_dict = make_dict(__old)

                keys = set.union(set(__old_dict.keys()), set(__new_dict.keys()))
                for key in keys:
                        a = Counter(__new_dict.get(key))
                        b = Counter(__old_dict.get(key))
                        sentiments = dict(a+b)
                        sentiments.update({"name": key})
                        aggregated_dict.append(sentiments)

                return aggregated_dict

        @staticmethod
        def flattening_dict(key, value):
                """
                key: ambience-overall 
                value: {'sentiment': [u'super-positive', u'super-positive', u'neutral']}

                Output: {"name": ambience-overall, u'super-positive': 2, u'neutral': 1}
                """
                __dict = dict(Counter(value.get("sentiment")))
                __dict.update({"name": key})
                return __dict


        @staticmethod
        def make_sentences_dict(noun_phrases):
                """
                Makes sentences_dict from self.c_sentences, self.predicted_sentiment, self.ambience_tags
                of the form
                { "ambience-null": {"sentences": [(__sent, __sentiment), (__sent, __sentiment), .. ],
                    "similar": None,
                    "sentiment": ["positive", "negative", "super-positive", ]},

                "decor": { }, }

                """
                sentences_dict = dict()
                for __sentiment, sent, __category in noun_phrases:
                        if not sentences_dict.has_key(__category):
                                sentences_dict.update({__category: {"sentiment": [__sentiment]}})

                        else:
                                sentiment = sentences_dict.get(__category).get("sentiment")
                                sentiment.append(__sentiment)

                                sentences_dict.update({
                                            __category: {"sentiment": sentiment,}})
                return sentences_dict



def convert_sentences(self, __object):
                return {"sentence": __object[0],
                        "sentiment": __object[1]}


        @print_execution
        def result_lambda(self, __dict):
                __dict.update({"sentences": map(self.convert_sentences, __dict.get("sentences"))})
                try:
                        i_likeness = "%.2f"%(float(__dict.get("positive")*100)/( __dict.get("negative") + __dict.get("positive")))
                except ZeroDivisionError:
                        i_likeness = '100'

                o_likeness =  "%.2f"%(float(__dict.get("positive")*self.total_positive + __dict.get("negative")*self.total_negative)/self.total)
                __dict.update({"i_likeness": i_likeness})
                __dict.update({"o_likeness": o_likeness})


 def make_result(self):
                self.total_positive = sum([__dict.get("positive") for __dict in  self.clustered_nps])
                self.total_negative = sum([__dict.get("negative") for __dict in  self.clustered_nps])
                self.total = self.total_positive + self.total_negative

                map(self.result_lambda, self.clustered_nps)
                final_result = sorted(self.clustered_nps, reverse= True,
                                            key=lambda x: x.get("negative")+ x.get("positive") + x.get("neutral"))

                return final_result



if __name__ == "__main__":
        ins = EachEatery(eatery_id="48016")
        
        for review_id, review_text in ins.return_non_processed_reviews()[0: 2]:
                per_review_instance = PerReview(review_id, review_text, "48016")
                per_review_instance.run()
