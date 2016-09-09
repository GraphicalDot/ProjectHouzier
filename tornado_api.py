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
from termcolor import cprint 
from pyfiglet import figlet_format

from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
from stanford_corenlp import save_tree
from connections import sentiment_classifier, tag_classifier

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
                                                        





class SentenceTokenization(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                """
                
                cprint(figlet_format("Now exceuting %s"%self.__class__.__name__, font='mini'), attrs=['bold'])

                text = self.get_argument("text")
                link = self.get_argument("link")
                tokenizer = None

                
                conll_extractor = ConllExtractor()
                topia_extractor = extract.TermExtractor()
                
                if link:
                        print "Link is present, so have to run goose to extract text"
                        print link 


                text =  text.replace("\n", "")
                
                if not tokenizer:
                        tokenizer = SentenceTokenizationOnRegexOnInterjections()
                        result = tokenizer.tokenize(text)
                else:
                        result = nltk.sent_tokenize(text)

                tags = TAG_CLASSIFIER_LIB.predict(result)
                sentiments = SENTI_CLASSIFIER_LIB_THREE_CATEGORIES.predict(result)


                def assign_proba(__list):
                        return {"mixed": round(__list[0], 2), 
                                "negative": round( __list[1], 2), 
                                "neutral": round(__list[2], 2) , 
                                "positive": round(__list[3], 2), }
                
                      
                sentiment_probabilities = map(assign_proba, SENTI_CLASSIFIER_LIB_THREE_CATEGORIES.predict_proba(result))

                new_result = list()
                
                
                for sentence, tag, sentiment, probability in zip(result, tags, sentiments, sentiment_probabilities):
                        try:
                                subcategory = list(eval('{0}_SB_TAG_CLASSIFIER_LIB.predict(["{1}"])'.format(tag[0:4].upper(), sentence)))[0]
                        except:
                                subcategory = None

                        if max(probability) < .7:
                                polarity_result = "can't decide"
                        else:
                                polarity_result = "decided"

                        file_name, dependencies, indexeddependencies = save_tree(sentence)

                        if file_name:
                                with open(file_name, "rb") as image_file:
                                        encoded_string = base64.b64encode(image_file.read())
                        else:
                                    encoded_string = None
        
                        blob = TextBlob(sentence)
                        tb_nps = list(blob.noun_phrases) 
                        
                        blob = TextBlob(sentence, np_extractor=conll_extractor)
                        tb_conll_nps = list(blob.noun_phrases) 

                        te_nps = [e[0] for e in topia_extractor(sentence)]

                        print sentence, dependencies, "\n" 
                        new_result.append(
                                {"sentence": sentence,
                                "encoded_string": encoded_string,
                                "polarity": sentiment, 
                                "sentiment_probabilities": probability, 
                                "dependencies": dependencies, 
                                "indexeddependencies": indexeddependencies,
                                "polarity_result": polarity_result,
                                "noun_phrases": ["a", "b", "c"],
                                "tag": tag, 
                                "tb_conll_nps": tb_conll_nps,
                                "te_nps": te_nps, 
                                "subcategory": subcategory
                                            })

                self.write({"success": True,
			        "error": False,
			        "result": new_result,
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
                    (r"/sentence_tokenization", SentenceTokenization),
                    ]
                settings = dict(cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",)
                tornado.web.Application.__init__(self, handlers, **settings)
                self.executor = ThreadPoolExecutor(max_workers=60)



if __name__ == '__main__':
    cprint(figlet_format('Server Reloaded', font='big'), attrs=['bold'])
    main()
