#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 5th February
Purpose: 
    This is the main file which loads all the classifiers into another folder at the same location named
    as InMemoryClassifiers

    Any edit to this file shall be written here

Edit One:
Author:
Purpose:

"""
import os
import sys
import inspect
from sklearn.externals import joblib 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from TagAnalysis import TagClassifier
from SentimentAnalysis import SentimentClassifier
from CostSubTagsAnalysis import CostSubTagClassifier



class LoadClassifiers:
        classifiers_list = ['all', 'logistic_regression_classifier', 'passive_agressive_classifier', 'perceptron_classifier', 'random_forests_classifier', 'ridge_regression_classifier', 'svm_classifier', 'svm_grid_search_classifier', 'svm_linear_kernel_classifier']
        def __init__(self, category, load=None, ):
                
                """
                Args:
                    load: default None which means load all classfiers in the InMemoryClassifiers folder which is present
                    in InMemoryClassifiers class MainAlgortihms module
                    else pass the name of the algorithm which should be revived with new dataset
                    till now the name of the algorithms which can be passed as an argument are 
                    'logistic_regression_classifier', 
                    'passive_agressive_classifier', 
                    'perceptron_classifier', 
                    'random_forests_classifier', nkita... Wish you a very Happy Birthday smile emoticon
                    'ridge_regression_classifier', 
                    'svm_classifier', 
                    'svm_grid_search_classifier', 
                    'svm_linear_kernel_classifier']

                """
                self.check_if_InMemoryClassifiers()

                self.load = ("all", load)[load != None]
                if category not in ["tag", "sentiment", "cost"]:
                        raise StandardError("The category you are trying to load into memory has not been implemented yet,\
                                Please choose from the following list %s"%["tag", "sentiment", "cost"] )
                
                if self.load not in self.classifiers_list:
                        raise StandardError("The classifier you are trying to load into memory has not been implemented yet,\
                                Please choose from the following list %s"%self.classifiers_list )



                self.category_classifiers_to_load = category
                

                print "{0}.load_{1}_classifiers()".format("self", self.category_classifiers_to_load)
                eval("{0}.load_{1}_classifiers()".format("self", self.category_classifiers_to_load)) 
                return

        def check_if_InMemoryClassifiers(self):
                """
                This method checks if InMemoryClassifiers directory exists or not
                """
                if not os.path.exists("InMemoryClassifiers"):
                        os.mkdir("InMemoryClassifiers")
                return


        def load_tag_classifiers(self):
                instance = TagClassifier()
                def load_classifiers(class_method, instance):
                        with cd("InMemoryClassifiers"):
                                print "{0}.{1}()".format("instance", class_method)
                                classifier = eval("{0}.{1}()".format("instance", class_method))
                                joblib_name_for_classifier = "{0}_tag.lib".format(class_method)
                                print classifier, joblib_name_for_classifier
                                joblib.dump(classifier, joblib_name_for_classifier) 
                        return


                if self.load == "all":
                        class_methods = [method[0] for method in inspect.getmembers(TagClassifier, predicate=inspect.ismethod) 
                            if method[0] not in ['loading_all_classifiers_in_memory', "__init__"]] 
            
                        for class_method in class_methods:
                                load_classifiers(class_method, instance)
                    
                        return

                load_classifiers(self.load, instance)
                return


        def load_sentiment_classifiers(self):
                instance = SentimentClassifier()
                def load_classifiers(class_method, instance):
                        with cd("InMemoryClassifiers"):
                                classifier = eval("{0}.{1}()".format("instance", class_method))
                                joblib_name_for_classifier = "{0}_sentiment.lib".format(class_method)
                                print classifier, joblib_name_for_classifier
                                joblib.dump(classifier, joblib_name_for_classifier) 
                        return


                if self.load == "all":
                        class_methods = [method[0] for method in inspect.getmembers(SentimentClassifier, predicate=inspect.ismethod) 
                            if method[0] not in ['loading_all_classifiers_in_memory', "__init__"]] 
            
                        for class_method in class_methods:
                                load_classifiers(class_method, instance)
                    
                        return

                load_classifiers(self.load, instance)
                return
        
        def load_cost_classifiers(self):
                """
                This method will be used to load cost sub tags
                """
                instance = CostSubTagClassifier()
                def load_classifiers(class_method, instance):
                        with cd("InMemoryClassifiers"):
                                classifier = eval("{0}.{1}()".format("instance", class_method))
                                joblib_name_for_classifier = "{0}_cost.lib".format(class_method)
                                print classifier, joblib_name_for_classifier
                                joblib.dump(classifier, joblib_name_for_classifier) 
                        return


                if self.load == "all":
                        class_methods = [method[0] for method in inspect.getmembers(SentimentClassifier, predicate=inspect.ismethod) 
                            if method[0] != "__init__"] 
            
                        for class_method in class_methods:
                                load_classifiers(class_method, instance)
                    
                        return

                load_classifiers(self.load, instance)
                return


class cd:
            def __init__(self, newPath):
                    self.newPath = newPath
            
            def __enter__(self):
                    self.savedPath = os.getcwd()
                    os.chdir(self.newPath)
            
            
            def __exit__(self, etype, value, traceback):
                os.chdir(self.savedPath)


if __name__ == "__main__":
        __ = LoadClassifiers(category="cost", load="svm_linear_kernel_classifier") 



