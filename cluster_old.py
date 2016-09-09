#!/usr/bin/env python

"""
Author: Kaali
Dated: 9 march, 2015
Purpose: This module deals with the clustering algorithms for the noun phrases taht has been extracted,
The purpose of the clustering is to find the similar noun phrases to stop their duplicacy for the end user

The Most appropriate algorithm to be implemented is k means clustering algorithm, but the problem with
this clustering algorithm is that it needs number of cluster in advance,
Kernel methods like liner kernel and rbf kernel can reduce the complexity of running time


The other algorithms to be implemented are, DBScan, heircharial clustering which removes the 
need to specify the number of cluster altoghther but are not so good.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import requests
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import Levenshtein
import codecs
import nltk
from compiler.ast import flatten
import time
from collections import Counter
TEST_DATA = [{u'frequency': 1, u'name': u'chocolate bath sort', u'polarity': 0},
            {u'frequency': 22, u'name': u'chocolate blood bath', u'polarity': 1},
            {u'frequency': 7, u'name': u'chocolate blood bath', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate blood bath dessert', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate blood bath dessert', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate bloodbath', u'polarity': 1},
            {u'frequency': 1, u'name': u'chocolate bloodbath', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate cake', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate desert', u'polarity': 1},
            {u'frequency': 2, u'name': u'chocolate dessert', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate dessert', u'polarity': 1},
            {u'frequency': 1, u'name': u'chocolate ice cream', u'polarity': 0},
            {u'frequency': 1, u'name': u'chocolate mousse', u'polarity': 0},
            {u'frequency': 2, u'name': u'chocolate mud bath', u'polarity': 1},]

TEST_DATA2 = [
     {u'frequency': 2, u'name': u'palak paneer', u'polarity': 1},
      {u'frequency': 1, u'name': u'palak paneer pudina masala', u'polarity': 1},
       {u'frequency': 1, u'name': u'palak pudina dish', u'polarity': 0},
        {u'frequency': 1, u'name': u'paneer bhujia', u'polarity': 0},
         {u'frequency': 1, u'name': u'paneer bhujia nawabi', u'polarity': 0},
          {u'frequency': 2, u'name': u'paneer dish', u'polarity': 0},
           {u'frequency': 1, u'name': u'paneer lababdaar', u'polarity': 0},
            {u'frequency': 5, u'name': u'paneer lababdar', u'polarity': 0},
             {u'frequency': 3, u'name': u'paneer lababdar', u'polarity': 1},
              {u'frequency': 1, u'name': u'paneer makhani', u'polarity': 0},
               {u'frequency': 4, u'name': u'paneer malai kofta', u'polarity': 0},
                {u'frequency': 1, u'name': u'paneer malai kofta', u'polarity': 1},
                 {u'frequency': 1, u'name': u'paneer mushroom', u'polarity': 0},
                  {u'frequency': 1, u'name': u'paneer pakeeza', u'polarity': 0},
                   {u'frequency': 1, u'name': u'paneer pakeeza', u'polarity': 1},
                    {u'frequency': 4, u'name': u'paneer tikka', u'polarity': 1},
                     {u'frequency': 4, u'name': u'paneer tikka', u'polarity': 0},
                      {u'frequency': 1, u'name': u'paneer tikka massala', u'polarity': 0},
                       {u'frequency': 1, u'name': u'paneer tikka.the chicken', u'polarity': 0},
                        {u'frequency': 1, u'name': u'panner tikka', u'polarity': 0},
                         {u'frequency': 1, u'name': u'papads', u'polarity': 0},
                          {u'frequency': 9, u'name': u'parikarma', u'polarity': 1},
                           {u'frequency': 1, u'name': u'parikarma', u'polarity': 0},
                            {u'frequency': 16, u'name': u'parikrama', u'polarity': 1},
                             {u'frequency': 10, u'name': u'parikrama', u'polarity': 0},
                              {u'frequency': 1, u'name': u'parikrama coz', u'polarity': 1},
                               {u'frequency': 1, u'name': u'parikrama drink', u'polarity': 1},
                                {u'frequency': 1, u'name': u'parikrama few years', u'polarity': 1},
                                 {u'frequency': 1, u'name': u'parikrama fruit punch dere', u'polarity': 1},
                                  {u'frequency': 2, u'name': u'parikrama restaurant', u'polarity': 1},
                                   {u'frequency': 1, u'name': u'parikrama restaurant', u'polarity': 0},
                                    {u'frequency': 4, u'name': u'pathetic food', u'polarity': 0},
                                     {u'frequency': 2, u'name': u'pathetic food', u'polarity': 1},
                                      {u'frequency': 2, u'name': u'pathetic mocktails', u'polarity': 0},
                                       {u'frequency': 2, u'name': u'pathetic taste', u'polarity': 0},
                                        {u'frequency': 1, u'name': u'peach cardinal', u'polarity': 0},
                                         {u'frequency': 1, u'name': u'peach cardinal', u'polarity': 1},
]






def merge_similar_elements(__result):
        """
        Merging noun phrases who have excat similar spellings with each other and return a dictionary in the form
        of
        {
        u'ice cream': {'data': [('negative', 3)]},
        u'ice cream chocolate': {'data': [('positive', 1)]},
        u'ice tea': {'data': [('positive', 6), ('negative', 5), ('negative', 1)]},
        u'iced tea': {'data': [('negative', 4), ('positive', 2), ('negative', 1)]},
        u'icelolly': {'data': [('negative', 1)]},
        }
        """
        
        without_similar_elements = dict()
        for i in __result:
                if without_similar_elements.get(i.get("name")):
                        without_similar_elements.get(i.get("name")).get("data").append((
                            ("negative" if i.get("polarity") == 0 else "positive", i.get("frequency")) 
                            )) 

                else:
                        without_similar_elements.update({i.get("name"): 
                                    {"data": [("negative" if i.get("polarity") == 0 else "positive", i.get("frequency"))],
                                    "similar": [],    
                                        }})


        return without_similar_elements

def Check_Levenshtein(__result):
        """
        Takes input from merge_similar_elements
        and then merge elements who have Levenshtein ratio greater than .8
        """
        __similar = dict()
        def yield_key():
                for key in __result.keys():
                        yield key
                
        for key_1 in yield_key():
                for key_2 in yield_key():
                        
                        if key_1 == key_2:
                            pass
                        
                        else:
                                st = 'Levenshtein.ratio("{1}", "{0}")'.format(key_1, key_2)
                                if eval(st) > .8:
                                        print key_1, key_2, "\n\n"
                                        try:
                                                new_data = __result[key_1]["data"]
                                                new_data.extend(__result[key_2]["data"])
                                                new_similar =  __result[key_1]["similar"]
                                                new_similar.append(key_2)
                                                new_dict = {key_1: {
                                                                "data": new_data,
                                                                "similar": new_similar,   
                                                            }}
                                                print new_dict
                                                __similar.update(new_dict)
                                                __result.pop(key_2)
                                        except Exception as e:
                                                print e
                                                pass
                                else:
                                        if __similar.get(key_2):
                                                pass
                                        else:
                                                __similar.update({key_2: __result[key_2]})
                                

        return __similar
             

def with_numpy(__result):
        """
        new_list = [[0, 1], [0, 5], [1, 5], [1, 9], [3, 7], [5, 9]]

        """
        X = np.zeros((len(__result), len(__result)), dtype=np.float)
        __keys = __result.keys()
       
        for element in enumerate(__keys):
                print element
        start = time.time()
        new_list = list()
        double_list = list()
        for i in xrange(0, len(__keys)):
                for j in xrange(0, len(__keys)):
                        st = 'Levenshtein.ratio("{1}", "{0}")'.format(__keys[i], __keys[j])
                        ratio = eval(st)
                        X[i][j] = ratio
                        if ratio >.8:
                            if i != j:
                                if [j ,i] not in new_list:
                                        new_list.append([i, j])
                            
                                        if not bool(double_list):
                                                double_list.append([i, j])
                                                #print "This is the double list %s"%double_list
                                                #print "The double list is empty"
                                        else:
                                                n = 0
                                                print "This is i = %s and this is j = %s"%(i, j)
                                                for __c in double_list:
                                                        if i in __c or j in __c:
                                                                l = double_list[n]
                                                                l.extend([i, j])
                                                                double_list[n] = l  
                                                                break
                                                        else:
                                                                double_list.append([i, j])
                                                                break
                                                        print double_list
                                                        print "\n\n"
                                                        n += 1
        print "time taken to make matrix is %s"%(time.time() - start)
        #print X
        print new_list
        print double_list
        clusters = [list(set(element))for element in double_list]
        without_clusters =  set.difference(set(range(0, len(__keys))), set(flatten(clusters)))
        print [__keys[__c] for __c in without_clusters], "\n\n"

        print "Now printing clusters"
        for __a in [[__keys[i] for i in __c] for __c in clusters]:
                print __a


class Clustering:

        def __init__(self, noun_phrases_list):
                """
                Args:
                    noun_phrases_list:
                            type: list of dicts with each dict in the form of
                            {u'frequency': 8, u'name': u'main course', u'polarity': 0},
                            with polarity = 0 implies negative sentiment
                            positive otherwise
                """
                __noun_phrases = [__dict.get("name") for __dict in noun_phrases_list]
                vector = TfidfVectorizer(ngram_range=(1, 2), min_df = 1, max_df = 1.0, decode_error = "ignore")
                tfidf = vector.fit_transform(__noun_phrases)
                self.X = (tfidf*tfidf.T).A

        def create_ratio_matrix(__result):
                start = time.time()
                X = np.zeros((len(__result), len(__result)), dtype=np.float)
                for i in xrange(0, len(__result)):
                        for j in xrange(0, len(__result)):
                                st = 'Levenshtein.ratio("{1}", "{0}")'.format(__result[i].get("name"), __result[j].get("name"))
                                X[i][j] = eval(st)
                print "time taken to make matrix is %s"%(time.time() - start)        
                return X

        def create_distance_matrix(__result):
                start = time.time()
                X = np.zeros((len(__result), len(__result)), dtype=np.float)
                for i in xrange(0, len(__result)):
                        for j in xrange(0, len(__result)):
                                st = 'Levenshtein.distance("{1}", "{0}")'.format(__result[i].get("name"), __result[j].get("name"))
                                X[i][j] = eval(st)
                print "time taken to make matrix is %s"%(time.time() - start)        
                return X


        @staticmethod
        def lev_dist(source, target):
            if source == target:
                return 0


            # Prepare a matrix
            slen, tlen = len(source), len(target)
            dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    
            for i in xrange(slen+1):
                    dist[i][0] = i
            for j in xrange(tlen+1):
                    dist[0][j] = j

            
            # Counting distance, here is my function
            for i in xrange(slen):
                    for j in xrange(tlen):
                            cost = 0 if source[i] == target[j] else 1
                            
                            """
                            if source[i] == " " or target[j]  == " ":
                                cost = 0
                            """
                            dist[i+1][j+1] = min(
                                    dist[i][j+1] + 1,   # deletion
                                    dist[i+1][j] + 1,   # insertion
                                    dist[i][j] + cost   # substitution
                            )
            return dist[-1][-1]



        def db_scan(self):
                """
                This is the db_scan mehtod fo the class Clustering
                """
                db_a = DBSCAN(eps=0.3, min_samples=5).fit(self.X)
                print db_a.labels_
        
        @staticmethod
        def k_means(*args, **kwargs):
                pass
        
        @staticmethod
        def mean_shift(noun_phrases_list):
                bandwidth = estimate_bandwidth(self.X, quantile=0.3) 
                __mean_shift= MeanShift(bandwidth=bandwidth)
                __mean_shift.fit(self.X)
                print __mean_shift.cluster_centers_

if __name__ == "__main__":
        """
        payload = {'category': 'food',
             'eatery_id': '3393',
              'ner_algorithm': 'stanford_ner',
               'pos_tagging_algorithm': 'hunpos_pos_tagger',
                'total_noun_phrases': 15,
                 'word_tokenization_algorithm': 'punkt_n_treebank'}

        r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
        result = merge_similar_elements(r.json()["result"])
        #print TEST_DATA, "\n\n"
        """
        #result = merge_similar_elements(TEST_DATA2)
        result = merge_similar_elements(TEST_DATA2)
        print result, "\n\n\n"

        print "Length of the result after creating dict out of it %s "%len(result.keys())
        print with_numpy(result)
        """
        test_data = [u'chocolate blood bath', u'chocolate blood bath', u'white chocolate', u'chocolate mud bath', u'much chocolate', 
                u'dessert choclolate blood bath', u'love chocolate', u'whole lot', u'hot chocolate ganache', 
                u'desserts favourite se blood bath', u'chocolate dessert', u"' chocolate blood bath", u'bloodbath', 
                u'death chocolate', u'love chocolate blood bath', u'chocolate wafer', u'chocolate cake', u'chocolate bloodbath', 
                u'chocolate bloodbath', u'chocolate blood bath dessert', u'chocolate bath sort', u'chocolate sauce', 
                u'miniature bubble bath', u'coz other food items', u'whole lot', u'layered chocolate cake', u'chocolate desert', 
                u'chocolate dessert', u'banana chocolate cronut', u'chocolate person', u'pthe chocolate blood bath', 
                u'last chocolate bath tub', u'chocolate blood bath dessert', u'white chocolate', u'chocolate mousse', 
                u'choc blud bath', u'chocolate ice cream', u'chicken pao bao']
        eatery_id = "4571"
        payload = {"eatery_id": eatery_id, "category":"food", "total_noun_phrases": 15, 
                "word_tokenization_algorithm": 'punkt_n_treebank', "pos_tagging_algorithm": "hunpos_pos_tagger", 
                "ner_algorithm": "stanford_ner"}
        r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
        Clustering.db_scan(r.json()["result"])
        #print Clustering.lev_dist('chin', 'clan')
        #print Levenshtein.distance('chocolate blood bath', 'chocolate blood bath dessert')
        for __str in  [(u'chocolate blood bath gadhe', u'blood bath')]:
                if __str[0] == __str[1]:
                        print 0
                        break

                if len(__str[0]) > 2 or len(__str[1]) > 2:
                        for i in __str[0].split():
                                for j in __str[1].split():
                                        st = 'Levenshtein.ratio("{1}", "{0}")'.format(i, j)
                                        print i, j
                                        print eval(st), "\n"

                else:
                        st = 'Levenshtein.ratio("{1}", "{0}")'.format(__str[0], __str[1])
                        print eval(st)

        def cal(__string):
            for __str in  [__string]:
                __sim = list()
                
                __ngram = min([len(__s) for __s in flatten([__str[0].split(" "), __str[1].split(" ")])])
                print "Number of ngrams %s"%__ngram
                for i in ["".join(__e) for __e in nltk.ngrams(__str[0], __ngram)]:
                        for j in ["".join(__e) for __e in nltk.ngrams(__str[1], __ngram)]:
                                        st = 'Levenshtein.ratio("{1}", "{0}")'.format(i, j)
                                        __sim.append(eval(st))
                __max = max(["".join(__e) for __e in nltk.ngrams(__str[0], __ngram)], ["".join(__e) for __e in nltk.ngrams(__str[1], __ngram)])
                __min = min(["".join(__e) for __e in nltk.ngrams(__str[0], __ngram)], ["".join(__e) for __e in nltk.ngrams(__str[1], __ngram)])
                print "Length of max ngram range %s and min ngram range %s"%(len(__max), len(__min))
                print sum(__sim)
                #print __sim
                print __sim.count(1)
                if len(__min)*5 < sum(__sim):
                            print "similar strings"
                else:
                            print "dissimilar strings"
            st = 'Levenshtein.ratio("{1}", "{0}")'.format(__string[0], __string[1])
            print "Levenshtein ratio between two strings is %s"%eval(st)
        
        def without_spaces_cal(__string):

                __sim = list()
                
                __ngram = min([len(__s) for __s in flatten([__string[0].split(" "), __string[1].split(" ")])])
                __str_1 = __string[0].replace(" ", "")
                __str_2 = __string[1].replace(" ", "")
                print "Number of ngrams %s"%__ngram
                for i in ["".join(__e) for __e in nltk.ngrams(__str_1, __ngram)]:
                        for j in ["".join(__e) for __e in nltk.ngrams(__str_2, __ngram)]:
                                        st = 'Levenshtein.ratio("{1}", "{0}")'.format(i, j)
                                        __sim.append(eval(st))
                __max = max(["".join(__e) for __e in nltk.ngrams(__str_1, __ngram)], ["".join(__e) for __e in nltk.ngrams(__str_2, __ngram)])
                __min = min(["".join(__e) for __e in nltk.ngrams(__str_1, __ngram)], ["".join(__e) for __e in nltk.ngrams(__str_2, __ngram)])
                print "Length of max ngram range %s and min ngram range %s"%(len(__max), len(__min))
                print sum(__sim)
                #print __sim
                print __sim.count(1)
                if len(__min)*5 < sum(__sim):
                            print "similar strings"
                else:
                            print "dissimilar strings"
                st = 'Levenshtein.ratio("{1}", "{0}")'.format(__str_1, __str_2)
                print "Levenshtein ratio between two strings is %s"%eval(st)

        test_data = [('chocolate blood bath', u'chocolate blood bath'), 
                ('white chocolate', u'chocolate mud bath'), 
                ('chocolate blood bath', u'much chocolate'),
                ('chocolate blood bath', u'dessert choclolate blood bath'),
                ('chocolate blood bath', 'love chocolate blood bath'), 
                ('chocolate blood bath', 'chocolate bath sort'),
                ('chocolate blood bath dessert', 'chocolate blood bath'), 
                ('chocolate blood bath', 'desserts favourite se blood bath'), 
                ('chocolate blood bath', 'banana chocolate cronut'), 
                ('chocolate blood bath', 'pthe chocolate blood bath'), 
                ('chocolate blood bath', u'chocolate dessert'), 
                ('chocolate blood bath', u'chocolate ice cream'), 
                ('chocolate blood bath', u'chicken pao bao'),
                ('chocolate blood bath', 'last chocolate bath tub'), 
                ('chocolate blood bath', 'layered chocolate cake'), 
                ('paneer tadka', 'paneer tikka'), 
                ('paneer bhurji', 'panaeer bhujia'), 
                ('mud bath', u'chocolate mud bath'), 

                ]
                u'dessert choclolate blood bath', u'love chocolate', u'whole lot', u'hot chocolate ganache', 
                u'desserts favourite se blood bath', u'chocolate dessert', u"' chocolate blood bath", u'bloodbath', 
                u'death chocolate', u'love chocolate blood bath', u'chocolate wafer', u'chocolate cake', u'chocolate bloodbath', 
                u'chocolate bloodbath', u'chocolate blood bath dessert', u'chocolate bath sort', u'chocolate sauce', 
                u'miniature bubble bath', u'coz other food items', u'whole lot', u'layered chocolate cake', u'chocolate desert', 
                u'chocolate dessert', u'banana chocolate cronut', u'chocolate person', u'pthe chocolate blood bath', 
                u'last chocolate bath tub', u'chocolate blood bath dessert', u'white chocolate', u'chocolate mousse', 
                u'choc blud bath', u'chocolate ice cream', u'chicken pao bao']
        short_test_data = [('chocolate blood bath', u'chocolate blood bath'), 
                ('chocolate blood bath', 'banana chocolate cronut'), 
                ('chocolate blood bath', 'pthe chocolate blood bath'), 
                ('paneer bhurji', 'panaeer bhujia'), 
                ('mud bath', u'chocolate mud bath'), 
                ]
        #for element in test_data:
        for element in test_data:
                print element
                print without_spaces_cal(element), "\n"



        """

