#!/usr/bin/env python
#-*- coding: utf-8 -*-
from In_Memory_Main_Classification import InMemoryMainClassifier, cd
import inspect
from sklearn.externals import joblib
from paths import path_in_memory_classifiers
import numpy

def get_all_algorithms_result(text="If only name of the algorithms are required", sentences_with_classification=True, if_names=False):
	"""
	This method will compare and delivers the accuracy of every algorithm computed in every class method which
	starts with with_
	To add any new algorithm just add that algortihm name starting with with__
	Args:
		sentences_with_classification: a list of dictionaries with sentence and its classfication
	"""
	
	print "entered into get all"
	results = list()

	print sentences_with_classification

	#tokenizer = SentenceTokenization()
	#new_data = tokenizer.tokenize(text)
	
	classifier_cls = InMemoryMainClassifier(["food", "ambience", "cost", "service", "overall", "null"])
	
	cls_methods_for_algortihms = [method[0] for method in inspect.getmembers(classifier_cls, predicate=inspect.ismethod) if method[0] not in ['loading_all_classifiers_in_memory', "__init__"]]
	
	if if_names:
		result = [cls_method for cls_method in cls_methods_for_algortihms]
		return result
		
		
	sentences, target = numpy.array(zip(*sentences_with_classification))

	for cls_method in cls_methods_for_algortihms:
		with cd(path_in_memory_classifiers):
			classifier = joblib.load("{0}_tag.lib".format(cls_method))
		predicted = classifier.predict(list(sentences))

		correct_classification = float(len([element for element in zip(predicted, target) if element[0] == element[1]]))

		print "{0} gives -- {1}".format(cls_method.replace("with_", ""), correct_classification/len(predicted))
		results.append({"algorithm_name": " ".join(cls_method.replace("with_", "").split("_")), 
				"accuracy": "{0:.2f}".format(correct_classification/len(predicted))})

	return results


