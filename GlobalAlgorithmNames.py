#!/usr/bin/env python
import os
file_name = os.path.dirname(os.path.abspath(__file__))
print file_name

path_in_memory_classifiers = "%s/Text_Processing/PrepareClassifiers/InMemoryClassifiers/newclassifiers"%file_name

print path_in_memory_classifiers
from sklearn.externals import joblib

TAGS = ["food", "service", "cost", "ambience", "null", "overall"]
FOOD_SUB_TAGS = ['dishes', 'menu-food', 'null-food', 'overall-food', 'place-food', 'sub-food']
COST_SUB_TAGS = ['cheap', 'cost-null', 'expensive', 'not worth', 'value for money']
SERV_SUB_TAGS = ['booking', 'management', 'presentation', 'service-null', 'service-overall',\
        'staff', 'waiting-hours']

AMBI_SUB_TAGS = [u'ambience-null', 'ambience-overall', 'crowd', 'dancefloor', 'decor', \
        'in-seating', 'music', 'open-area', 'romantic', 'smoking-zone', 'sports', 'sports-screens', 'view', 'sports-props']


SENTIMENT_TAGS = ["super-positive", "super-negative", "neutral", "positive", "negative"]


TAG_CLASSIFIER = "svm_tag_classifier_latest.lib"
SENTI_CLASSIFIER_THREE_CATEGORIES = "svm_linear_kernel_sentiment_new_categories_classifier.lib"



FOOD_SB_TAG_CLASSIFIER = "food_classifier_latest"
COST_SB_TAG_CLASSIFIER = "cost_classifier_latest"
SERV_SB_TAG_CLASSIFIER = "service_classifier_latest"
AMBI_SB_TAG_CLASSIFIER =  "ambience_classifier_latest"
NOUN_PHSE_ALGORITHM_NAME = "topia"

TAG_CLASSIFY_ALG_NME = TAG_CLASSIFIER.replace(".lib", "")
FOOD_SB_CLSSFY_ALG_NME = FOOD_SB_TAG_CLASSIFIER.replace(".lib", "")
SERV_SB_CLSSFY_ALG_NME = SERV_SB_TAG_CLASSIFIER.replace(".lib", "")
AMBI_SB_CLSSFY_ALG_NME = AMBI_SB_TAG_CLASSIFIER.replace(".lib", "")
COST_SB_CLSSFY_ALG_NME = COST_SB_TAG_CLASSIFIER.replace(".lib", "")

TAG_CLASSIFIER_LIB = joblib.load("{0}/{1}".format(path_in_memory_classifiers, TAG_CLASSIFIER)) 
SENTI_CLASSIFIER_LIB_THREE_CATEGORIES = joblib.load("{0}/{1}".format(path_in_memory_classifiers, SENTI_CLASSIFIER_THREE_CATEGORIES)) 



FOOD_SB_TAG_CLASSIFIER_LIB = joblib.load("{0}/{1}".format(path_in_memory_classifiers, FOOD_SB_TAG_CLASSIFIER))
COST_SB_TAG_CLASSIFIER_LIB = joblib.load("{0}/{1}".format(path_in_memory_classifiers, COST_SB_TAG_CLASSIFIER))
SERV_SB_TAG_CLASSIFIER_LIB = joblib.load("{0}/{1}".format(path_in_memory_classifiers, SERV_SB_TAG_CLASSIFIER))
AMBI_SB_TAG_CLASSIFIER_LIB = joblib.load("{0}/{1}".format(path_in_memory_classifiers, AMBI_SB_TAG_CLASSIFIER))
