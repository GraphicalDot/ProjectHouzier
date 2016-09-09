#!/usr/bin/env python


import pymongo
import ConfigParser
from sklearn.externals import joblib
import os
from topia.termextract import extract 
import jsonrpclib
from simplejson import loads
from elasticsearch import Elasticsearch, helpers
from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
sentence_tokenizer = SentenceTokenizationOnRegexOnInterjections()
noun_phrase_extractor = extract.TermExtractor()


this_file_path = os.path.dirname(os.path.abspath(__file__))

config = ConfigParser.RawConfigParser()
config.read("variables.cfg")

ELASTICSEARCH_IP = config.get("elasticsearch", "ip")

path_for_classifiers = "%s/Text_Processing/PrepareClassifiers/InMemoryClassifiers/newclassifiers"%(this_file_path) 

sentiment_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "sentiment_classification_library")))
tag_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "tag_classification_library")))
food_sb_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "food_algorithm_library")))
ambience_sb_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "ambience_algorithm_library")))
service_sb_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "service_algorithm_library")))
cost_sb_classifier = joblib.load("%s/%s"%(path_for_classifiers, config.get("algorithms", "cost_algorithm_library")))
                         


data_db_connection = pymongo.MongoClient(config.get("dataDB", "ip"), config.getint("dataDB", "port"))
data_db = data_db_connection[config.get("dataDB", "database")]

reviews = data_db[config.get("dataDB", "reviews")]
eateries = data_db[config.get("dataDB", "eateries")]


result_db_connection = pymongo.MongoClient(config.get("resultsDB", "ip"), config.getint("resultsDB", "port"))
result_db  = result_db_connection[config.get("resultsDB", "database")]
reviews_results_collection = result_db[config.get("resultsDB", "review_result")]
eateries_results_collection = result_db[config.get("resultsDB", "eatery_result")]
discarded_nps_collection=  result_db[config.get("resultsDB", "discarded_nps")]
short_eatery_result_collection = result_db[config.get("resultsDB", "short_eatery_result")]
area_collection = result_db[config.get("resultsDB", "area")]
print area_collection

users_db_connection = pymongo.MongoClient(config.get("usersDB", "ip"), config.getint("usersDB", "port"))
users_db  = users_db_connection[config.get("usersDB", "database")]
users_reviews_collection = users_db[config.get("usersDB", "usersreviews")]
users_feedback_collection = users_db[config.get("usersDB", "usersfeedback")]
users_details_collection = users_db[config.get("usersDB", "usersdetails")]
users_queried_addresses_collection = users_db[config.get("usersDB", "usersqueriedaddresses")]
users_dish_collection = users_db[config.get("usersDB", "usersdishcollection")]


pictures_connection = pymongo.MongoClient(config.get("picturesDB", "ip"), config.getint("picturesDB", "port"))
pictures_db  = pictures_connection[config.get("picturesDB", "database")]
pictures_collection = pictures_db[config.get("picturesDB", "collection")]


try:
        corenlpserver = jsonrpclib.Server("http://{0}:{1}".format(config.get("corenlpserver", "ip"), config.getint("corenlpserver", "port")))
        loads(corenlpserver.parse("Testing corenlp server."))
except Exception as e:
        raise StandardError("Corenlp Server is not running, Please run corenlp server  %s at port %s"%(config.get("corenlpserver", "ip"), config.getint("corenlpserver", "port")))



ES_CLIENT = Elasticsearch(ELASTICSEARCH_IP, timeout=30)

server_address = "{0}://{1}:{2}".format(config.get("server", "protocol"), config.get("server", "ip"), config.getint("server", "port"))




class SolveEncoding(object):
        def __init__(self):
                pass


        @staticmethod
        def preserve_ascii(obj):
                if not isinstance(obj, unicode):
                        obj = unicode(obj)
                obj = obj.encode("ascii", "xmlcharrefreplace")
                return obj

        @staticmethod
        def to_unicode_or_bust(obj, encoding='utf-8'):
                if isinstance(obj, basestring):
                        if not isinstance(obj, unicode):
                                obj = unicode(obj, encoding)
                return obj


class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        RESET='\033[0m'


