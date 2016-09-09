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
from heuristic_clustering import HeuristicClustering

    
from sklearn.externals import joblib
from collections import Counter


class FoodWordCloudApiHelper:
        def __init__(self, **kwargs):
                allowed_kwargs = ['reviews', 'eatery_name', 'category', 'total_noun_phrases', 'word_tokenization_algorithm_name', 
                        'noun_phrases_algorithm_name', 'pos_tagging_algorithm_name', 'tag_analysis_algorithm_name', 
                        'sentiment_analysis_algorithm_name', 'np_clustering_algorithm_name', 'ner_algorithm_name', 'with_celery', 
                        "do_sub_classification"]
                self.__dict__.update(kwargs)
                for kwarg in allowed_kwargs:
                        assert eval("self.{0}".format(kwarg)) != None
                
                self.tag_classifier = joblib.load("{0}/{1}".format(path_in_memory_classifiers, self.tag_analysis_algorithm_name))              
                
                #self.sub_tag_classifier = joblib.load("{0}/{1}".format(path_in_memory_classifiers, "svm_linear_kernel_classifier_food_sub_tags.lib"))
                self.sub_tag_classifier = joblib.load("{0}/{1}".format(path_in_memory_classifiers, 
                    "svm_linear_kernel_classifier_food_sub_tags_8May.lib"))
                
                self.sentiment_classifier = joblib.load("{0}/{1}".format(path_in_memory_classifiers,\
                                                       "svm_linear_kernel_classifier_sentiment_new_dataset_30April.lib"))              
                
                self.sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()

                self.clustered_nps = list()
                self.normalized_sent_sentiment_nps = list()
        
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
        
        
        def run(self):
                """
                It returns the result
                """


                self.sent_tokenize_reviews() #Tokenize reviews, makes self.reviews_ids, self.sentences
                self.predict_tags()          #Predict tags, makes self.predict_tags
                        
                self.filtered_list = [e for e in zip(self.review_ids, self.sentences, self.predicted_tags) 
                                                                                if e[2] == self.category]

                if self.do_sub_classification:
                        """
                        Classify food sentences into furthur these categories
                        'dishes', 'food-null', 'menu-food', 'null-food', 'overall-food', 'place-food', 'sub-food'
                        """
                        self.review_ids, self.sentences, self.predicted_tags = zip(*self.filtered_list)
                        self.food_sub_tag_classification()
                        self.filtered_list = [e for e in zip(self.review_ids, self.sentences, self.predicted_sub_tags) 
                                                                                if e[2] == "dishes"]


                self.c_review_ids, self.c_sentences, self.c_predicted_tags = zip(*self.filtered_list)
                
                self.predict_sentiment() #makes self.predicted_sentiment
                 
                self.extract_noun_phrases() #makes self.noun_phrases
                self.normalize_sentiments() #makes self.normalized_noun_phrases
                
                self.do_clustering() #makes self.clustered_nps
                self.result = self.make_result()
                

        #@print_execution
        def food_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.predicted_sub_tags = self.sub_tag_classifier.predict(self.sentences)
                return self.predicted_sub_tags




        #@print_execution
        def sent_tokenize_reviews(self):
                """
                Tokenize self.reviews tuples of the form (review_id, review) to sentences of the form (review_id, sentence)
                and generates two lists self.review_ids and self.sentences
                """
                sentences = list()
                for review in self.reviews:
                        for __sentence in self.sent_tokenizer.tokenize(review[1]):
                                        __sentence = SolveEncoding.preserve_ascii(__sentence)
                                        sentences.append([review[0], __sentence])

                self.review_ids, self.sentences = zip(*sentences)
                return 
                        
        #@print_execution
        def predict_tags(self):
                """
                Predict tags of the sentence which were being generated by self.sent_tokenize_reviews
                """
                self.predicted_tags = self.tag_classifier.predict(self.sentences)
                return self.predicted_tags

        #@print_execution
        def predict_sentiment(self):
                """
                Predict sentiment of self.c_sentences which were made by filtering self.sentences accoring to 
                the specified category
                """
                self.c_predicted_sentiment = self.sentiment_classifier.predict(self.c_sentences)
                return 
        

        #@print_execution
        def extract_noun_phrases(self):
                """
                Extarct Noun phrases for the self.c_sentences for each sentence and outputs a list 
                self.sent_sentiment_nps which is of the form 
                [('the only good part was the coke , thankfully it was outsourced ', 
                                            u'positive', [u'good part']), ...]
                """
                self.noun_phrases_algorithm_name = "topia"
                __nouns = NounPhrases(self.c_sentences, default_np_extractor=self.noun_phrases_algorithm_name)

                self.sent_sentiment_nps = [__tuple for __tuple in 
                        zip(self.c_sentences, self.c_predicted_sentiment, __nouns.noun_phrases[self.noun_phrases_algorithm_name])
                        if __tuple[2]]

                return self.sent_sentiment_nps 
        
        #@print_execution
        def normalize_sentiments(self, ignore_super=False):
                """
                self.sent_sentiment_nps = 
                [('the only good part was the coke , thankfully it was outsourced ', 
                                            u'positive', [u'good part']), 
                ("I had the bset ferror rocher shake ever", "super-positive", "ferror rocher shake", "positive"), ...]
                Now, the above  list has super-negative and super-positive sentiments associated with
                them,
                        ignore_super:
                                if True:
                                        super-positive and super-negative will be treated same as positive and negative
                                else:
                                        super-positive will consider as two positives,
                
                for element in self.noun_phrases:
                        if element[0].startswith("super"):
                                self.normalized_noun_phrases.append((element[1], element[0].split("-")[1]))
                                if not ignore_super:
                                        self.normalized_noun_phrases.append((element[1], element[0].split("-")[1]))
                        else:
                                self.normalized_noun_phrases.append((element[1], element[0]))

                return self.normalized_noun_phrases
                """
                for (sentence, sentiment, noun_phrases) in self.sent_sentiment_nps:
                        __nouns = list()
                        if sentiment.startswith("super"):
                                sentiment = sentiment.split("-")[1]
                                __nouns.extend(noun_phrases)
                                if not ignore_super:
                                        __nouns.extend(noun_phrases)
                        else:
                                __nouns.extend(noun_phrases)
                        self.normalized_sent_sentiment_nps.append([sentence, sentiment, __nouns ])
                
                return self.normalized_sent_sentiment_nps


        #@print_execution
        def do_clustering(self):
                """
                Deals with clusteing by import another module for food, HeuristicClustering
                passes on self.normalized_sent_sentiment_nps which is of the form 
                Input:
                    [('the only good part was the coke , thankfully it was outsourced ', 
                                            u'positive', [u'good part']), ...]
                Output:
                        makes a class variable self.clustered_nps from output HeuristicClustering 
                        [{"name": "ferror rocher shake", "positive": 20, "negative": 10, "neutral": 3,
                            "similar": ["ferror rocher", "i like ferror rocher", "luv ferror rocher",], 
                            "sentences": [("i luv ferror rocher shake", "positive"), 
                                        ("I went there specially for ferror rocher skae", "neutral"), ..]}, ...]
                """
            
                __result = HeuristicClustering(self.normalized_sent_sentiment_nps, self.c_sentences, self.eatery_name)
                self.clustered_nps = sorted(__result.result, reverse=True, key= lambda x: x.get("positive")+x.get("negative"))
                return self.clustered_nps


        #@print_execution
        def convert_sentences(self, __object):
                return {"sentence": __object[0],
                        "sentiment": __object[1]}
                                

        #@print_execution
        def result_lambda(self, __dict):
                __dict.update({"sentences": map(self.convert_sentences, __dict.get("sentences"))})
                try:
                        i_likeness = "%.2f"%(float(__dict.get("positive")*100)/( __dict.get("negative") + __dict.get("positive")))
                except ZeroDivisionError:
                        i_likeness = '100'

                o_likeness =  "%.2f"%(float(__dict.get("positive")*self.total_positive + __dict.get("negative")*self.total_negative)/self.total)
                __dict.update({"i_likeness": i_likeness})
                __dict.update({"o_likeness": o_likeness})
                
                
        #@print_execution
        def make_result(self):
                self.total_positive = sum([__dict.get("positive") for __dict in  self.clustered_nps])
                self.total_negative = sum([__dict.get("negative") for __dict in  self.clustered_nps])
                self.total = self.total_positive + self.total_negative
                
                map(self.result_lambda, self.clustered_nps)
                final_result = sorted(self.clustered_nps, reverse= True, 
                                            key=lambda x: x.get("negative")+ x.get("positive") + x.get("neutral"))
                
                return final_result



                """
                word_tokenize = WordTokenize(sentences,  default_word_tokenizer= word_tokenization_algorithm_name)
                word_tokenization_algorithm_result = word_tokenize.word_tokenized_list.get(word_tokenization_algorithm_name)


                __pos_tagger = PosTaggers(word_tokenization_algorithm_result,  default_pos_tagger=pos_tagging_algorithm_name)
                pos_tagging_algorithm_result =  __pos_tagger.pos_tagged_sentences.get(pos_tagging_algorithm_name)


                __noun_phrases = NounPhrases(pos_tagging_algorithm_result, default_np_extractor=noun_phrases_algorithm_name)
                noun_phrases_algorithm_result =  __noun_phrases.noun_phrases.get(noun_phrases_algorithm_name)

                #result = [element for element in zip(predicted_sentiment, noun_phrases_algorithm_result) if element[1]]
                result = [element for element in __result if element[1]]
                """

