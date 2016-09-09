#!/usr/bin/env python
#-*- coding: utf-8 -*-

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy

class Sklearn_RandomForest:
	"""
	Source Wikipedia:

	Ensembles combine multiple hypotheses to form a (hopefully) better hypothesis. In other words, an ensemble is a technique for 
	combining many weak learners in an attempt to produce a strong learner
	
	Evaluating the prediction of an ensemble typically requires more computation than evaluating the prediction of a single model, 
	so ensembles may be thought of as a way to compensate for poor learning algorithms by performing a lot of extra computation. 
	Fast algorithms such as decision trees are commonly used with ensembles (for example Random Forest), although slower algorithms 
	can benefit from ensemble techniques as well.

	An ensemble is itself a supervised learning algorithm, because it can be trained and then used to make predictions. The trained 
	ensemble, therefore, represents a single hypothesis. This hypothesis, however, is not necessarily contained within the hypothesis 
	space of the models from which it is built. Thus, ensembles can be shown to have more flexibility in the functions they can 
	represent. This flexibility can, in theory, enable them to over-fit the training data more than a single model would, but in 
	practice, some ensemble techniques (especially bagging) tend to reduce problems related to over-fitting of the training data.

	Empirically, ensembles tend to yield better results when there is a significant diversity among the models.Many ensemble methods, 
	therefore, seek to promote diversity among the models they combine.[6][7] Although perhaps non-intuitive, more random algorithms 
	(like random decision trees) can be used to produce a stronger ensemble than very deliberate algorithms (like entropy-reducing 
	decision trees)
	
	RANDOM FORESTS are an ensemble learning method for classification (and regression) that operate by constructing a multitude of 
	decision trees at training time and outputting the class that is the mode of the classes output by individual trees.


	Other sources: 
		http://citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics/http://citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics/
		
	The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners. 
	Given a training set X = x1, …, xn with responses Y = y1 through yn, bagging repeatedly selects a bootstrap sample of the training 
	set and fits trees to these samples:
	
	For b = 1 through B:
		1.Sample, with replacement, n training examples from X, Y; call these Xb, Yb.
		2.Train a decision or regression tree fb on Xb, Yb.
			After training, predictions for unseen samples x' can be made by averaging the predictions from all the individual 
			regression trees on x':
			
			\hat{f} = \frac{1}{B} \sum_{b=1}^B \hat{f}_b (x')

	or by taking the majority vote in the case of decision trees.
	In the above algorithm, B is a free parameter. Typically, a few hundred to several thousand trees are used, depending on the size 
	and nature of the training set. Increasing the number of trees tends to decrease the variance of the model, without increasing the 
	bias. As a result, the training and test error tend to level off after some number of trees have been fit. An optimal number of 
	trees B can be found using cross-validation, or by observing the out-of-bag error: the mean prediction error on each training 
	sample xᵢ, using only the trees that did not have xᵢ in their bootstrap sample.[9]
	
	
	The above procedure describes the original bagging algorithm for trees. Random forests differ in only one way from this general 
	scheme: they use a modified tree learning algorithm that selects, at each candidate split in the learning process, a random subset 
	of the features. The reason for doing this is the correlation of the trees in an ordinary bootstrap sample: if one or a few features 
	are very strong predictors for the response variable (target output), these features will be selected in many of the B trees, 
	causing them to become correlated.
	
	Typically, for a dataset with p features, √p features are used in each split.
	
	"""
	def __init__(self, data, target):
		self.data = data
		self.target = target

	def classifier(self):
		#count_vect = CountVectorizer()
		#X_train_counts = count_vect.fit_transform(self.data)
		#tfidf_transformer = TfidfTransformer()
		#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		"""
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
		X_train = vectorizer.fit_transform(self.data)
		RandomForest_classifier = RandomForestClassifier(n_estimators=10)

		RandomForest_classifier.fit(X_train.toarray(), self.target)
		"""

		classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
				('clf', RandomForestClassifier(n_estimators=10)),])

		classifier.fit(self.data, self.target)
		return classifier




	def predict_with_chi_test(self, data_to_predict):
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
		X_train = vectorizer.fit_transform(self.data)

		ch2 = SelectKBest(chi2, k="all")
		X_train = ch2.fit_transform(X_train, self.target)

		X_test = vectorizer.transform(data_to_predict)
		X_test = ch2.transform(X_test)

		RandomForest_classifier = RandomForestClassifier(n_estimators=10)

		RandomForest_classifier.fit(X_train.toarray(), self.target)

		prediction = RandomForest_classifier.predict(X_test.toarray())
		return zip(data_to_predict, prediction)



