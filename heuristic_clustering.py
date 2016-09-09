#!/usr/bin/env python

"""
Author: Kaali
Dated: 9 march, 2015
Purpose: This module deals with the clustering of the noun phrases, Evverything it uses are heuristic rules because
till now i am unable to find any good clutering algorithms which suits our needs.
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



class HeuristicClustering:
        def __init__(self, __result, __sentences, __eatery_name):
                """
                Args:
                    __result
                            type: list of dictionaries
                            example: [{'positive': 20, 'name': u'teppanyaki grill', 'negative': 12}, 
                            {'positive': 8, 'negative': 10, 'name': u'main course'}, {'positive': 7, 'negative': 8, 'name': u'kylin'}]
                """
                
                if __eatery_name:
                        self.list_to_exclude = flatten(["food", "service", "cost", "ambience", "place", "Place", 
                            "great", "good", __eatery_name.lower().split(), "rs"])
                        #self.list_to_exclude = ["food", "service", "cost", "ambience", "delhi", "Delhi", 
                        #       "place", "Place", __eatery_name.lower().split()]
                else:
                        self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", "Delhi", "place", "Place"]
                
                self.list_to_exclude = flatten(self.list_to_exclude)
                self.data = __result

                self.sentences = __sentences
                self.NERs = self.ner()
                
                print "Length of the old data after exclusion %s"%len(self.data)
                
                self.new_data = self.merge_similar_elements()
                self.keys = self.new_data.keys()
                print "Length of the new data after merging similar elements  %s"%len(self.keys)
                self.clusters = list()
                self.result = list()
                
                self.filter_clusters()
                self.without_clusters =  set.difference(set(range(0, len(self.keys))), set(flatten(self.clusters)))
        
                self.populate_result()
                self.result = sorted(self.result, reverse=True, key= lambda x: x.get("positive") + x.get("negative"))


        def ner(self):
                __list = list()
                for sent in self.sentences:
                        tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent.encode("ascii", "xmlcharrefreplace"))))
                        for subtree in tree.subtrees(filter = lambda t: t.label()=='GPE'):
                                __list.append(" ".join([e[0] for e in subtree.leaves()]).lower())
                
                print __list
                return __list



        def merge_similar_elements(self):
                """
                Merging noun phrases who have exact similar spellings with each other and return a dictionary in the form
                u'ice tea': {'positive', 6, 'negative': 5},
                u'iced tea': {'positive', 2, 'negative', 10},
                u'icelolly': {'positive': 0, 'negative', 1},
                }
                """
                
                without_similar_elements = dict()
                for i in self.data:
                        """
                        if i.get("name") in list(set(self.NERs)):
                                print "This noun_phrase belongs to ner {0}".format(i.get("name"))
                                pass
                        """
                        if bool(set.intersection(set(i.get("name").split(" ")),  set(self.list_to_exclude))):
                                pass    

                        elif without_similar_elements.get(i.get("name")):
                                result = without_similar_elements.get(i.get("name"))
                                polarity = "negative" if i.get("polarity") == 0 else "positive"
                        
                                if polarity == "negative":
                                        new_frequency_negative = result.get("negative") + i.get("frequency")
                                else:
                                        new_frequency_negative = result.get("negative")


                                if polarity == "positive":
                                        new_frequency_positive = result.get("positive") + i.get("frequency")
                                else:
                                        new_frequency_positive = result.get("positive")
                                
                                without_similar_elements.update(
                                    {i.get("name"): 
                                        {"negative": new_frequency_negative,
                                        "positive": new_frequency_positive,
                                        }})

                
                        else:
                                without_similar_elements.update(
                                    {i.get("name"): 
                                        {"negative" if i.get("polarity") == 0 else "positive": i.get("frequency"),
                                        "negative" if i.get("polarity") == 1 else "positive": 0,
                                        }})

                return without_similar_elements

        def filter_clusters(self):
            """

            """
            X = np.zeros((len(self.new_data), len(self.new_data)), dtype=np.float)
      
             
            new_list = list()
            
            for i in xrange(0, len(self.keys)):
                    for j in xrange(0, len(self.keys)):
                            if X[i][j] == 0:
                                    st = 'Levenshtein.ratio("{1}", "{0}")'.format(self.keys[i], self.keys[j])
                                    ratio = eval(st)
                                    X[i][j] = ratio
                                    X[j][i] = ratio
            
            
            #Making tuples of the indexes for the element in X where the rtion is greater than .76
            indices = np.where(X > .76)
            new_list = zip(indices[0], indices[1])
            found = False
            for e in new_list:
                    for j in self.clusters:
                            if bool(set.intersection(set(e), set(j))):
                                    j.extend(e)
                                    found = True
                                    break
                    if not found:    
                            self.clusters.append(list(e))
                            found = False
                    found = False

            #Removing duplicate elements from clusters list
            self.clusters = [list(set(element)) for element in self.clusters if len(element)> 2]
            """
            for element in self.clusters:
                    print [self.keys[key] for key in element]
            """
            return 

        def populate_result(self):
        
                for __int_key in self.without_clusters:
                        new_dict = dict()
                        name = self.keys[__int_key]
                        new_dict = self.new_data[name]
                        new_dict.update({"name": name})
                        self.result.append(new_dict)

                for cluster_list in self.clusters:
                        __dict = self.maximum_frequency(cluster_list)
                        if __dict:
                                self.result.append(__dict)

                return

        def maximum_frequency(self, cluster_list):
                """
                Returning name with maximum frequency in a cluster, by joining all the frequencies
                """
                """
                if set.intersection(set([self.keys[__e] for __e in cluster_list]), set(self.NERs)):
                        print "Match found"
                        print set.intersection(set([self.keys[__e] for __e in cluster_list]), set(self.NERs))
                        print cluster_list
                        return False
                """
                result = list()
                positive, negative = int(), int()
                positive_name, negative_name = str(), str()
                print [self.keys[element] for element in cluster_list]
                for element in cluster_list:
                        name = self.keys[element]    
                        new_dict = self.new_data[name]
                        new_dict.update({"name": name})
                        result.append(new_dict)        
                        positive = positive +  self.new_data[name].get("positive") 
                        negative = negative +  self.new_data[name].get("negative") 
        
        
                result = sorted(result, reverse= True, key=lambda x: x.get("positive"))
                print "THE name chosen is %s"%result[0].get("name"), "\n"
                return {"name": result[0].get("name"), "positive": positive, "negative": negative}


        def check_if_shortform(self, str1, str2):
                """
                To identify if "bbq nation" is similar to "barbeque nation"

                """
                if bool(set.intersection(set(str1.split()), set(str2.split()))):

                        if set.issubset(set(str1), set(str2)):
                                return True

                        if set.issubset(set(str2), set(str1)):
                                return 

                return False

if __name__ == "__main__":
        """
        def transform(__object): 
                __l = __object[1] 
                __l.append(__object[0]) 
                return list(set(__l))

        __dict = {} 
        for e in __list:
                __dict.setdefault(min(e), []).append(max(e))
        
        return map(transform, __dict.keys())

        
        
        payload = {'category': 'food',
             'eatery_id': '301489',
              'ner_algorithm': 'stanford_ner',
               'pos_tagging_algorithm': 'hunpos_pos_tagger',
                'total_noun_phrases': 15,
                 'word_tokenization_algorithm': 'punkt_n_treebank'}

        r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
        result = r.json()["result"]
        """
        result = [{u'polarity': 1, u'frequency': 84, u'name': u'main course'},
                {u'polarity': 0, u'frequency': 1, u'name': u'main course order'},
                {u'polarity': 1, u'frequency': 2, u'name': u'other subway outlets'},
                {u'polarity': 1, u'frequency': 2, u'name': u'best subway outlet'},
                {u'polarity': 1, u'frequency': 1, u'name': u'good subway outlet'},
                {u'polarity': 1, u'frequency': 1, u'name': u'other subway'},
                
                {u'polarity': 0, u'frequency': 7, u'name': u'dal makhani'},
                {u'polarity': 0, u'frequency': 5, u'name': u'dal makhni'},
                {u'polarity': 1, u'frequency': 2, u'name': u'love dal makhni'},
                {u'polarity': 1, u'frequency': 2, u'name': u'daal makhni'},
                {u'polarity': 1, u'frequency': 1, u'name': u'daal makhani creamy'},
                {u'polarity': 1, u'frequency': 1, u'name': u'daal punjabi'},
                {u'polarity': 1, u'frequency': 1, u'name': u'paneer lababdaar'},
                {u'polarity': 1, u'frequency': 1, u'name': u"' dal makhani"},
                {u'polarity': 0, u'frequency': 1, u'name': u'dal makhni'},
                {u'polarity': 0, u'frequency': 1, u'name': u'dal makahani'},
                {u'polarity': 0, u'frequency': 1, u'name': u'daal makhani'},


                {u'polarity': 0, u'frequency': 5, u'name': u'paneer lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'murg lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'pannneer lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'paaner lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'panner laabaabdar'},


                {u'polarity': 1, u'frequency': 3, u'name': u'parikrama restaurant'},
                {u'polarity': 1, u'frequency': 3, u'name': u'parikarma restaurant'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama drink'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama hotel'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama coz'},
                
                



                {u'polarity': 0, u'frequency': 60, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 26, u'name': u'barbeque nation'},
        {u'polarity': 0, u'frequency': 20, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 20, u'name': u'bbq nation'},
        {u'polarity': 0, u'frequency': 14, u'name': u'barbeque nation'},
        {u'polarity': 1, u'frequency': 13, u'name': u'good food'},
        {u'polarity': 1, u'frequency': 13, u'name': u'ice cream'},
        {u'polarity': 1, u'frequency': 12, u'name': u'chicken biryani'},
        {u'polarity': 1, u'frequency': 12, u'name': u'barbecue nation'},
        {u'polarity': 1, u'frequency': 10, u'name': u'great food'},
        {u'polarity': 1, u'frequency': 9, u'name': u'chicken tikka'},
        {u'polarity': 1, u'frequency': 9, u'name': u'good place'},
        {u'polarity': 1, u'frequency': 84, u'name': u'main course'},
        {u'polarity': 0, u'frequency': 60, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 26, u'name': u'barbeque nation'},
        {u'polarity': 0, u'frequency': 20, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 20, u'name': u'bbq nation'},
        {u'polarity': 0, u'frequency': 14, u'name': u'barbeque nation'},
        {u'polarity': 1, u'frequency': 13, u'name': u'good food'},
        {u'polarity': 1, u'frequency': 13, u'name': u'ice cream'},
        {u'polarity': 1, u'frequency': 12, u'name': u'chicken biryani'},
        {u'polarity': 1, u'frequency': 12, u'name': u'barbecue nation'},
        {u'polarity': 1, u'frequency': 10, u'name': u'great food'},
        {u'polarity': 1, u'frequency': 9, u'name': u'chicken tikka'},
        {u'polarity': 1, u'frequency': 9, u'name': u'good place'},
        {u'polarity': 1, u'frequency': 9, u'name': u'paneer tikka'},
        {u'polarity': 1, u'frequency': 8, u'name': u'tasty food'},
        {u'polarity': 1, u'frequency': 8, u'name': u'gulab jamun'},
        {u'polarity': 0, u'frequency': 7, u'name': u'gulab jamun'},
        {u'polarity': 0, u'frequency': 7, u'name': u'chicken tikka'},
        {u'polarity': 0, u'frequency': 7, u'name': u'barbecue nation'},
        {u'polarity': 0, u'frequency': 7, u'name': u'bbq nation'},
        {u'polarity': 1, u'frequency': 4, u'name': u'dal makhani'},
        {u'polarity': 1, u'frequency': 4, u'name': u'main course everything'},
        {u'polarity': 1, u'frequency': 3, u'name': u'main course section'},
        {u'polarity': 0, u'frequency': 2, u'name': u'bt main course'},
        {u'polarity': 1, u'frequency': 2, u'name': u'main coursenot'},
        {u'polarity': 1, u'frequency': 2, u'name': u'fabulous main course'}]


        """
        m = HeuristicClustering(result, 'Barbeque Nation')
        for element in m.result:
                print element
        for __dict1 in result:
            for __dict2 in result:
                    #if not __dict1.get("name") != __dict2.get("name"):
                            st = 'Levenshtein.ratio("{1}", "{0}")'.format(__dict1.get("name"), __dict2.get("name"))
                            print eval(st), "\t", __dict1.get("name"), __dict2.get("name")
        """


        test_data = [['non veg buffet', 'veg buffet', False],
        ['mutton seekh', 'mutton seekh kababs', True],
        ['mutton seekh', 'mutton seekh kabab', True],
        ['mutton dum biryani', 'delicious mutton biryani', True],
        ['main course', 'fabulous main course', True],
        ['other bar nations', 'barbeque nation', True],
        ['hot gulab jamun', 'da gulab jamun', True],
        ['main course menu', 'main course', True],
        ['unlimited tasty food', 'unlimited food', True],
        ['coconut brownie', 'chocolate brownie', False],
        ['neat experience', 'bad experience', True],
        ['barbeque nation team', 'barbeque nation buffet', False],
        ['chocolate cake', 'chocolate sauce', False],
        ['great sauce', 'great place', False],
        ['kadhai chicken', 'khatta chicken', False],
        ['kadhai chicken', 'chicken', False],
        ['kadhai chicken', 'peshawari chicken leg', False],
        ['kadhai chicken', 'teriyaki chicken', False],
        ['kadhai chicken', 'fish prawn chicken', False],
        ['kadhai chicken', 'teriyaki chicken', False],
        ['kadhai chicken', 'jerk chicken', False],
        ['kadhai chicken', 'chicken', False],
        ['bbq nation', 'barbque nation', True],
        ['gud mutton seekh kbab', 'mutton seekh kebab', True]] 
        
        def longest_common_string(str1, str2):
                break_strings = lambda __str: flatten([[e[0:i+1] for i in range(0, len(e))] for e in __str.split(" ")])
               
                common_strings = len(list(set.intersection(set(break_strings(str1)), set(break_strings(str2)))))

                min_length = min(sum([len(e) for e in str1.split(" ")]), sum([len(e) for e in str2.split(" ")]))
                max_length = max(sum([len(e) for e in str1.split(" ")]), sum([len(e) for e in str2.split(" ")]))
                if min_length == min([len(e) for e in flatten([str1.split(" "), str2.split(" ")])]):
                        
				return float(common_strings)/max_length
                return float(common_strings)/min_length

      
        for __e in test_data:
                st = 'Levenshtein.ratio("{1}", "{0}")'.format(__e[0], __e[1])
                print __e[0], "\t", __e[1], __e[2], "\t", True if longest_common_string(__e[0], __e[1]) > .75 else False

        result = [[5, 10, 5, 11, 5, 24, 5, 28, 10, 5, 10, 11, 10, 24, 10, 28, 11, 5, 11, 10, 11, 24, 11, 28, 
                24, 5, 24, 10, 24, 11, 24, 28, 28, 5, 28, 10, 28, 11, 28, 24], [6, 9, 6, 13, 6, 14, 6, 23, 
                6, 30, 9, 6, 9, 13, 9, 14, 9, 23, 9, 30, 13, 6, 13, 9, 13, 14, 13, 23, 13, 30, 14, 6, 14, 9, 14, 
                13, 14, 23, 14, 30, 23, 6, 23, 9, 23, 13, 23, 14, 23, 30, 30, 6, 30, 9, 30, 13, 30, 14, 30, 23], 
                [15, 22, 15, 34, 22, 15, 34, 15], [31, 32, 32, 31]]

        __list = [[10, 5], [11, 5], [24, 5], [28, 5], [9, 6], [13, 6], [14, 6], [23, 6], [30, 6], [6, 9], 
                [13, 9], [14, 9], [23, 9], [30, 9], [5, 10], [11, 10], [24, 10], [28, 10], [5, 11], [10, 11], 
                [24, 11], [28, 11], [6, 13], [9, 13], [14, 13], [23, 13], [30, 13], [6, 14], [9, 14], [13, 14], 
                [23, 14], [30, 14], [22, 15], [34, 15], [15, 22], [6, 23], [9, 23], [13, 23], [14, 23], [30, 23], 
                [5, 24], [10, 24], [11, 24], [28, 24], [5, 28], [10, 28], [11, 28], [24, 28], [6, 30], [9, 30], 
                [13, 30], [14, 30], [23, 30], [32, 31], [31, 32], [15, 34]]

        [[u'philadelphia rolls', 'Philadelphia rolls', u'philadelphia', u'veg philadelphia rolls'], [u'amazing gastronomic experience', 'gastronomic experience'], ['nut chocolate', 'chocolate', u'chunky nut chocolate'], ['Yuzu Miso Sauce', 'Japanese miso soup', 'Japanese Yuzu miso sauce', u'yuzu miso sauce', u'japanese yuzu miso sauce'], [u'fried cream spring rolls', u'kylin special fried ice cream spring rolls', 'spring rolls', 'cream spring rolls', 'Fried Ice cream spring rolls', 'ice cream spring rolls', u'fried ice cream spring rolls'], ['Chicken Teriyaki', u'chicken teriyaki', u'chicken tepiniyaki dish', u'chicken teppanyaki'], ['Italian Smooch', u'italian smooch', ':- 1. italian smooch'], ['Spicy Salmon roll', 'salmon rolls', u'spicy salmon rolls', 'salmon maki', u'spicy salmon roll', 'Salmon Roll', u'salmon roll', u'spicy salmon maki'], [u'mango breeze drink', 'Mango breeze', u'mango breeze', 'mango Breeze drink'], ['Japanese Restaurant', u'authentic japanese restaurant', u'japanese restaurant', u'japanese 5 star restaurants', 'Japanese 5 star Restaurants', 'Japanese restaurant']]
