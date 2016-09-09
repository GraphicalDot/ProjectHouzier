#!/usr/bin/env python
#-*- coding: utf-8 -*-
#concatenating multiple feature extraction methods
#http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py
"""
http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html
http://stackoverflow.com/questions/17794313/unexpected-results-when-using-scikit-learns-svm-classification-algorithm-rbf-k?rq=1
http://scikit-learn.org/0.13/auto_examples/grid_search_digits.html#example-grid-search-digits-py
http://scikit-learn.org/stable/auto_examples/grid_search_digits.html
http://scikit-learn.org/0.13/modules/grid_search.html
http://scikit-learn.org/0.13/auto_examples/grid_search_digits.html#example-grid-search-digits-py
http://scikit-learn.org/0.13/auto_examples/grid_search_text_feature_extraction.html#example-grid-search-text-feature-extraction-py
http://stackoverflow.com/questions/23815938/recursive-feature-elimination-and-grid-search-using-scikit-learn?rq=1
http://stackoverflow.com/questions/14866228/combining-grid-search-and-cross-validation-in-scikit-learn?rq=1
http://stackoverflow.com/questions/15254243/different-accuracy-for-libsvm-and-scikit-learn?rq=1

Some good refrences to get a good background of the Natrual language processing algorithms

http://www.quora.com/What-are-the-advantages-of-different-classification-algorithms


difference between l1, l2 and elasticnet
http://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization
Two popular regularization methods are L1 and L2. If you're familiar with Bayesian statistics: 
L1 usually corresponds to setting a Laplacean prior on the regression coefficients - and picking a maximum a posteriori hypothesis. 
L2 similarly corresponds to Gaussian prior. As one moves away from zero, the probability for such a coefficient grows progressively smaller.

Feature unions 
http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

"""

import pymongo
from Algortihms import  SVMWithGridSearch
from Sentence_Tokenization import SentenceTokenizationOnRegexOnInterjections, CopiedSentenceTokenizer
import random
import itertools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import SGDClassifier                                                                                                            
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
import openpyxl

from sklearn.feature_selection import SelectKBest, chi2
from MainAlgorithms import path_in_memory_classifiers, path_trainers_file, path_parent_dir
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.lda import LDA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

connection = pymongo.Connection()
db = connection.modified_canworks
reviews = db.review


def print_term_frequencies(corpus):
        """
        Corpus shall be in the form of 
        corpus = ["This is very strange",
                          "This is very nice"]

        """
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        print dict(zip(vectorizer.get_feature_names(), idf))






def generate_test_data_3():
        __test_data = list()
        wb = openpyxl.load_workbook("/home/k/Programs/python/canworks/new_test_data.xlsx")
        sh = wb.get_active_sheet()
        test_data = [[cell.value for cell in r] for r in sh.rows]
        for element in test_data[1:]:
                if element[3] != "mix":
                        try:
                            __test_data.append((element[2].lower(), element[3]))
                        except Exception as e:
                            print e
                            pass
        return __test_data



def generate_test_data_sentiment():
        __test_data = list()
        wb = openpyxl.load_workbook("/home/k/Programs/python/canworks/new_test_data.xlsx")
        sh = wb.get_active_sheet()
        test_data = [[cell.value for cell in r] for r in sh.rows]
        for element in test_data[1:]:
                if element[4]:
                        try:
                                __test_data.append((element[2].lower(), element[4]))
                        except Exception as e:
                            print e
                            pass
        return __test_data


def with_svm_countvectorizer():
        pca = PCA()
        selection = SelectKBest()


        #classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=1)), ('tfidf', TfidfTransformer()),
        classifier = Pipeline([ ('vect', TfidfVectorizer(ngram_range=(1, 2), min_df=1,)),
            ('chi2', SelectKBest(chi2, k=900)),
            ('clf', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1e-3, n_iter=5)),])   

        tuned_parameters = {"clf__alpha": 1e-05,
                            "clf__n_iter": 50,
                            "clf__penalty": 'l1',
                            "tfidf__norm": 'l1',
                            "tfidf__use_idf": False,
                            "vect__max_df": 0.5,
                            "vect__max_features": None,
                            "vect__ngram_range": (1, 1),
                            }

        classifier.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 200 samples with SVM %.3f"%(classifier.score(TEST_SENTENCES, TEST_TARGET))



def sgd_with_grid_search():
        """
        Stochastic gradient descent
        """
        def search_params():
                """
                This function makes a gris search for suitable parametrs
                From the previous grid search these are the following tuned params which we got
                tuned_parameters = {"clf__n_iter": 80, 
                            clf__penalty: 'l1',
                            tfidf__norm: 'l1',
                            tfidf__use_idf: True,
                            vect__analyzer: 'char_wb',
                            vect__max_df: 1.0,
                            vect__max_features: None,
                            vect__ngram_range: (1, 4),
                            }
                """
    
    
                pipeline = Pipeline([('vect', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SGDClassifier()),
                                        ])


                parameters = { 'vect__max_df': (0.5, 0.75, 1.0),
                                'vect__max_features': (None, 50),
                                'vect__ngram_range': [(1,4)],  # unigrams or bigrams
                                'vect__analyzer': ['word', 'char', 'char_wb'],
                                'tfidf__use_idf': (True, False),
                                'tfidf__norm': ('l1', 'l2'),
                                'clf__alpha': (0.00001, 0.000001),
                                'clf__penalty': ('l1', 'elasticnet'),
                                'clf__n_iter': (10, 50, 80),
                                }

                classifier= GridSearchCV(pipeline, parameters, verbose=1)
        
                print "Best score: %0.3f" % classifier.best_score_
                print "Best parameters set:"
                best_parameters = classifier.best_estimator_.get_params()
                for param_name in sorted(parameters.keys()):
                        print "\t%s: %r" % (param_name, best_parameters[param_name])
                return
        
        
        classifier = Pipeline([('vect', CountVectorizer(max_df = 1.0, max_features =None, ngram_range=(1, 5), analyzer='char_wb')),
                                        ('tfidf', TfidfTransformer(use_idf=True, norm="l1")),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SGDClassifier(n_iter=80, penalty="l1", alpha=0.000001)),
                                        ])
        
        classifier.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 200 samples with sgd grid search %.3f"%(classifier.score(TEST_SENTENCES, TEST_TARGET))
        print "Accuracy with 500 samples with sgd grid search %.3f"%(classifier.score(TEST_SENTENCES_500, TEST_TARGET_500))
        
        classifier = Pipeline([('vect', CountVectorizer(max_df = 1.0, max_features =None, ngram_range=(1, 5), analyzer='char_wb')),
                                        ('tfidf', TfidfTransformer(use_idf=True, norm="l1")),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SGDClassifier(n_iter=80, penalty="l1", alpha=0.000001)),
                                        ])
        classifier.fit(SENTIMENT_TRAINING_SENTENCES, SENTIMENT_TARGETS)
        print "Accuracy with 200 samples with sgd grid search %.3f"%(classifier.score(SENTIMENT_TEST_SENTENCES, SENTIMENT_TEST_TARGET))




def with_svm():
        
        training_sentences, training_target_tags = return_tags_training_set()

        instance = SVMWithGridSearch(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        classifier = instance.classifier()


        print "Accuracy with 200 samples with SVM grid search %.3f"%(classifier.score(TEST_SENTENCES, TEST_TARGET))
        classifier= GridSearchCV(pipeline, tuned_parameters, verbose=1)



        classifier.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 200 samples with LDA %.3f"%(classifier.score(TEST_SENTENCES, TEST_TARGET))
        print "Best score: %0.3f" % classifier.best_score_
        print "Best parameters set:"
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
                print "\t%s: %r" % (param_name, best_parameters[param_name])

        


def with_support_vector_machines():
        """
        C :
                The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
                For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job 
                of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer 
                to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, 
                you should get misclassified examples, often even if your training data is linearly separable.

        """
                                        
        pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range=(1, 4), analyzer="char_wb")),
                                    ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SVC(C=1, kernel="linear", gamma=.0001)),
                                        ])

        """
        parameters = {
                        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1,4)],  # unigrams or bigrams
                        'vect__analyzer': ['word', 'char', 'char_wb'],
                        'clf__kernel': ['linear', 'rbf',], 
                        'clf__gamma': [1e-3, 1e-4],
                        'clf__C': [1, 10, 100, 1000]
                                 }



        classifier= GridSearchCV(pipeline, parameters, verbose=1)
        """


        
        pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range=(2, 6), analyzer="char_wb")),
                                    ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SVC(C=1, kernel="linear", gamma=.001)),
                                        ])
        pipeline.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 500 samples with svm search %.3f"%(pipeline.score(TEST_SENTENCES_500, TEST_TARGET_500))
        pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range=(2, 6), analyzer="char_wb")),
                                    ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SVC(C=1, kernel="linear", gamma=.01)),
                                        ])
        pipeline.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 500 samples with svm search %.3f"%(pipeline.score(TEST_SENTENCES_500, TEST_TARGET_500))
        
        pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range=(1, 6), analyzer="char_wb")),
                                    ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SVC(C=1, kernel="linear", gamma=0.0)),
                                        ])
        pipeline.fit(TAGS_TRAINING_SENTENCES, TAG_TARGETS)
        print "Accuracy with 500 samples with svm search %.3f"%(pipeline.score(TEST_SENTENCES_500, TEST_TARGET_500))
        """
        print "Best score: %0.3f" % classifier.best_score_
        print "Best parameters set:"
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
                print "\t%s: %r" % (param_name, best_parameters[param_name])

        """

def with_support_vector_machines_for_sentiment():
        """
        C :
                The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
                For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job 
                of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer 
                to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, 
                you should get misclassified examples, often even if your training data is linearly separable.

        """
                                        
        pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range=(2, 4), analyzer="char_wb")),
                                    ('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
                                        ('clf', SVC(C=1, kernel="linear", gamma=.001)),
                                        ])

        pipeline.fit(SENTIMENT_TRAINING_SENTENCES, SENTIMENT_TARGETS)

        print "Accuracy with 500 samples with SVM %.3f"%(pipeline.score(SENTIMENT_TEST_SENTENCES, SENTIMENT_TEST_TARGET))
        
        """
        print "Best score: %0.3f" % classifier.best_score_
        print "Best parameters set:"
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
                print "\t%s: %r" % (param_name, best_parameters[param_name])

        """

if __name__ == "__main__":
        TAGS_TRAINING_SENTENCES, TAG_TARGETS = return_tags_training_set()
        #SENTIMENT_TRAINING_SENTENCES, SENTIMENT_TARGETS = return_sentiment_training_set()
        SENTIMENT_TRAINING_SENTENCES, SENTIMENT_TARGETS = generate_training_sentiment_data_from_files()

        #TEST_SENTENCES, TEST_TARGET = zip(*generate_test_data())
        TEST_SENTENCES_500, TEST_TARGET_500 = zip(*generate_test_data_3())

        SENTIMENT_TEST_SENTENCES, SENTIMENT_TEST_TARGET = zip(*generate_test_data_sentiment())
        #with_svm_countvectorizer()
        #:with_svm()
        #with_lda()
        ##sgd_with_grid_search()
        with_support_vector_machines()
        #with_support_vector_machines_for_sentiment()
