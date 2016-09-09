#!/usr/bin/env python
#-*- coding: utf-8 -*-


from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy

class SVMWithGridSearch:
	def __init__(self, data, target):
		self.data = data
		self.target = target

	def classifier(self):
		#count_vect = CountVectorizer()
		#X_train_counts = count_vect.fit_transform(self.data)
		#tfidf_transformer = TfidfTransformer()
		#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		
		pipeline = Pipeline([ ('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
                                        ('chi2', SelectKBest(chi2, k="all")),
					('clf', SGDClassifier()),
					])

		# uncommenting more parameters will give better exploring power but will
		# increase processing time in a combinatorial way
                """
                gives around 50% for [(2, 3)]
                66% for [(1, 3)]
		68% for 'vect__ngram_range': [(1, 1), (1, 4), ],

                """

		parameters = { 'vect__max_df': (0.5, 0.75, 1.0),
				'vect__max_features': (None, 500, 1000),
				'vect__ngram_range': [(1, 1), (1, 2), (1, 3), ],  # unigrams or bigrams
				#'tfidf__use_idf': (True, False),
				#'tfidf__norm': ('l1', 'l2'),
				#'clf__alpha': (0.00001, 0.000001),
				'clf__penalty': ('l1', 'elasticnet'),
				#'clf__n_iter': (10, 50, 80),
				}


		svm_with_grid_search = GridSearchCV(pipeline, parameters, n_jobs=-2, verbose=1)
		svm_with_grid_search.fit(self.data, self.target)

		return svm_with_grid_search
