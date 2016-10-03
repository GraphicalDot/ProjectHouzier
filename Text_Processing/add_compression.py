#!/usr/bin/pypy

from os.path import dirname, abspath
from PreProcessingText import PreProcessText
from Vectorization import HouzierVectorizer
from Transformation import  HouzierTfIdf
from TrainingData.MongoData import TrainingMongoData
from nltk.stem import SnowballStemmer
from configs import base_dir, cd
from sklearn.decomposition import PCA, RandomizedPCA, NMF, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from sklearn.externals import joblib
from  CoreNLPScripts import  CoreNLPScripts
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
#on mac print option +# to search throught he file
from sklearn import preprocessing
import warnings
import os 
warnings.filterwarnings("ignore")
import time 
import numpy as np
from sklearn.metrics import accuracy_score
from cPickle import dump, load, HIGHEST_PROTOCOL

from configs import SentimentClassifiersPath, TagClassifiersPath,\
                FoodClassifiersPath, ServiceClassifiersPath,\
                AmbienceClassifiersPath, CostClassifiersPath 


from configs import SentimentVocabularyFileName, SentimentFeatureFileName, SentimentClassifierFileName
from configs import TagVocabularyFileName, TagFeatureFileName, TagClassifierFileName
from configs import FoodVocabularyFileName, FoodFeatureFileName, FoodClassifierFileName
from configs import ServiceVocabularyFileName, ServiceFeatureFileName, ServiceClassifierFileName
from configs import CostVocabularyFileName, CostFeatureFileName, CostClassifierFileName
from configs import AmbienceVocabularyFileName, AmbienceFeatureFileName, AmbienceClassifierFileName
import gzip

def add_compression(name, file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):

                with cd(file_path):
                        print "loading vectorizer"
                        vectorizer = joblib.load(file_name_vectorizer)
                        print "loading features"
                        feature_reduction_class = joblib.load(file_name_features)
                        print "loading classifier"
                        classifier = joblib.load(file_name_classifier)


                new_file_name_vectorizer = "%s.gz"%file_name_vectorizer
                new_file_name_features = "%s.gz"%file_name_features
                new_file_name_classifier = "%s.gz"%file_name_classifier

                new_data_path = "%s/%s"%(file_path, name)
                print new_file_name_features
                print new_file_name_vectorizer
                print new_file_name_classifier
                print new_data_path


                if not os.path.exists(new_data_path):
                        with cd(file_path):
                                    os.makedirs(name)

                with cd("%s/%s"%(file_path, name)):
                        print "Gzipping vectorizer"
                        with gzip.GzipFile(new_file_name_vectorizer, 'wb') as f:
                                joblib.dump(vectorizer, f, compress=('zlib', 9))
                        print "Gzipping vectorizer Completed"
                        
                        print "Gzipping features"
                        with gzip.GzipFile(new_file_name_features, 'wb') as f:
                                joblib.dump(feature_reduction_class, f, compress=('zlib', 9))
                        print "Gzipping features Completed"
                        
                        print "Gzipping classifier"
                        with gzip.GzipFile(new_file_name_classifier, 'wb') as f:
                                joblib.dump(classifier, f,  compress=('zlib', 9))
                        print "Gzipping classifier completed "




                        #feature_reduction_class=load(open(file_name_features, 'rb'))
                        #classifier= load(open(file_name_classifier, 'rb'))
                
                
                return 
         

if __name__ == "__main__":
        """
        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_after_corenlp_analysis(),
                                     "svmlk_sentiment_corenlp_classifier.pkl",
                                     "lk_vectorizer_sentiment_corenlp.pkl", 
                                    "sentiment_corenlp_features.pkl")
        data = TrainingMongoData.tag_data()
        TagClassifiers.svm_bagclassifier(data,
                                     "svmlk_tag_classifier.pkl",
                                     "lk_vectorizer_tag.pkl",
                                     "tag_features.pkl")

        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_three_categories(),
                                               SentimentClassifierFileName,
                                               SentimentVocabularyFileName, 
                                               SentimentFeatureFileName)

        """
        
        add_compression("SentimentClassifiers",
                                        SentimentClassifierFileName,
                                        SentimentVocabularyFileName, 
                                        SentimentFeatureFileName, 
                                        SentimentClassifiersPath)
        """
        add_compression("TagClassifiers",
                                        TagClassifierFileName,
                                         TagVocabularyFileName, 
                                         TagFeatureFileName, 
                                        TagClassifiersPath)
        

        add_compression("CostClassifiers",
                                        CostClassifierFileName,
                                        CostVocabularyFileName, 
                                        CostFeatureFileName, 
                                        CostClassifiersPath)

        add_compression("FoodClassifiers", 
                                        FoodClassifierFileName,
                                          FoodVocabularyFileName, 
                                          FoodFeatureFileName, 
                                          FoodClassifiersPath
                                          )
        add_compression("ServiceClassifiers", 
                                            ServiceClassifierFileName,
                                            ServiceVocabularyFileName, 
                                             ServiceFeatureFileName, 
                                            ServiceClassifiersPath)

        add_compression("AmbienceClassifiers",
                                                AmbienceClassifierFileName,
                                              AmbienceVocabularyFileName,
                                              AmbienceFeatureFileName,
                                              AmbienceClassifiersPath)
        
        """



