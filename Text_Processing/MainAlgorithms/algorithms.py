#!/usr/bin/env python
#-*- coding: utf-8 -*-

import nltk
import numpy
import random
import sys
import os
import time
from optparse import OptionParser
import inspect
import itertools
import numpy as np
import pymongo


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.externals import joblib
from sklearn.svm import SVC 

from paths import path_in_memory_classifiers, path_trainers_file, path_parent_dir



#Changing path to the main directory where all the modules are lying
sys.path.append(os.path.join(path_parent_dir))
from Sentence_Tokenization import SentenceTokenizationOnRegexOnInterjections, CopiedSentenceTokenizer
from Algortihms import Sklearn_RandomForest
from Algortihms import SVMWithGridSearch
from colored_print import bcolors


def timeit(method):
        def timed(*args, **kw):
                ts = time.time()
                result = method(*args, **kw)
                te = time.time()

                print '%s%r (%r, %r) %2.2f sec %s'%(bcolors.OKGREEN, method.__name__, args, kw, te-ts, bcolors.RESET)
                return result
        return timed



class Algortihms(object):
	
        def __init__(self):
                pass






	    @timeit
	    def svm_classifier(self):
		        classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
			            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])

		classifier.fit(self.training_sentences, self.training_target_tags)
		return classifier
	@timeit
	def logistic_regression_classifier(self):
		classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
			('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5)),])

		classifier.fit(self.training_sentences, self.training_target_tags)
		return classifier

	@timeit
	def perceptron_classifier(self):
		classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
			('clf', Perceptron(n_iter=50)),])
		
		classifier.fit(self.training_sentences, self.training_target_tags)
		return classifier


	@timeit
	def ridge_regression_classifier(self):
		classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
			('clf', RidgeClassifier(tol=1e-2, solver="lsqr")),])

		
		classifier.fit(self.training_sentences, self.training_target_tags)
		return classifier

	@timeit
	def passive_agressive_classifier(self):
		classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
			('clf', PassiveAggressiveClassifier(n_iter=50)),])

		classifier.fit(self.training_sentences, self.training_target_tags)
		return classifier

        """
        Could not be implemented beacuse it cant be run on sparse numpy array
	@timeit
	def random_forests_classifier(self):

		print "\n Running {0} \n".format(inspect.stack()[0][3])
		instance = Sklearn_RandomForest(self.training_sentences, self.training_target_tags)
		classifier = instance.classifier()
		

		return classifier
        """
	
	@timeit
	def svm_grid_search_classifier(self):
		print "\n Running {0} \n".format(inspect.stack()[0][3])
		instance = SVMWithGridSearch(self.training_sentences, self.training_target_tags)
		classifier = instance.classifier()
		return classifier
	
        
    @timeit
	def svm_grid_search_classifier(self):
		print "\n Running {0} \n".format(inspect.stack()[0][3])
		instance = SVMWithGridSearch(self.training_sentences, self.training_target_tags)
		classifier = instance.classifier()
		return classifier

        
    @timeit
	def svm_linear_kernel_classifier(self):
        print "\n Running {0} \n".format(inspect.stack()[0][3])
		classifier = Pipeline([ ('vect', CountVectorizer(ngram_range=(1, 6), analyzer="char_wb")),
                    ('tfidf', TfidfTransformer()),
                    ('chi2', SelectKBest(chi2, k="all")),
                    ('clf', SVC(C=1, kernel="linear", gamma=.0001)),])     
            
            
            
	    classifier.fit(self.training_sentences, self.training_target_tags)  
		return classifier


    @timeit
	def svm_linear_kernel_classifier(self):
        sklearn.base.TransformerMixin¶
		class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

        print "\n Running {0} \n".format(inspect.stack()[0][3])
		classifier = Pipeline([ ('vect', CountVectorizer(ngram_range=(1, 6), analyzer="char_wb")),
                    ('tfidf', TfidfTransformer()),
                    ('chi2', SelectKBest(chi2, k="all")),
                    ('clf', SVC(C=1, kernel="linear", gamma=.0001)),])     
            
            
            
	    classifier.fit(self.training_sentences, self.training_target_tags)  
		return classifier






class cd:
	def __init__(self, newPath):
		self.newPath = newPath

	def __enter__(self):
		self.savedPath = os.getcwd()
		os.chdir(self.newPath)
		
	def __exit__(self, etype, value, traceback):
		os.chdir(self.savedPath)


if __name__ == "__main__":
        tag_list = ["food", "ambience", "cost", "service", "overall", "null"]
        ins = InMemoryMainClassifier(tag_list)  
