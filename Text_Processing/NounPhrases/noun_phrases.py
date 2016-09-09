#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 31 january, 2015
This file lists the sentences and the noun phrasesthat hould be extracted
and test several noun phrases extraction algorithms whether they are providing desired output

Another method

train_sents = [
    [('select', 'VB'), ('the', 'DT'), ('files', 'NNS')],
        [('use', 'VB'), ('the', 'DT'), ('select', 'JJ'), ('function', 'NN'), ('on', 'IN'), ('the', 'DT'), ('sockets', 'NNS')],
            [('the', 'DT'), ('select', 'NN'), ('files', 'NNS')],
            ]


tagger = nltk.TrigramTagger(train_sents, backoff=default_tagger)
Note, you can use NLTK's NGramTagger to train a tagger using an arbitrarily high number of n-grams, but typically you don't get much performance 
increase after trigrams.
grammer = r"CustomNounP:{<JJ|VB|FW>?<NN.*>*<NN.*>}"
grammer = r"CustomNounP:{<JJ|VB|FW|VBN>?<NN.*>*<NN.*>}"

                food:
                <NNP><NNS>, "Mozralle fingers"
                (u'Chicken', u'NNP'), (u'Skewer', u'NNP'), (u'bbq', u'NN'), (u'Sauce', u'NN')
                (u'Mozarella', u'NNP'), (u'Fingers', u'NNP')
                 review_ids = ['4971051', '3948891', '5767031', '6444939', '6500757', '854440']
                '4971051' 
                     (u'Ferrero', u'NNP'), (u'Rocher', u'NNP'), (u'shake', u'NN'), 
                     (u'lemon', u'JJ'), (u'iced', u'JJ'), (u'tea', u'NN'), 
                     (u'mezze', u'NN'), (u'platter', u'NN'), 
                     (u'banoffee', u'NN'), (u'cronut', u'NN'),
                '3948891', 
                    (u'China', u'NNP'), (u'Box', u'NNP'), (u'with', u'IN'), (u'Chilly', u'NNP'), (u'Paneer', u'NNP'), 
                    (u'Vada', u'NNP'), (u'pao', u'NNP'), 
                    (u'Mezze', u'NNP'), (u'Platter', u'NNP'), 
                    (u'Naga', u'NNP'), (u'Chili', u'NNP'), (u'Toast', u'NNP'), 
                    (u'Paneer', u'NNP'), (u'Makhani', u'NNP'), (u'Biryani', u'NNP'), 
                    (u'Kit', u'NN'), (u'Kat', u'NN'), (u'shake', u'NN'), 
                    (u'ferrero', u'NN'), (u'rocher', u'NN'), (u'shake', u'NN'), 
                    
                '5767031', 
                     (u'Tennessee', u'NNP'), (u'Chicken', u'NNP'), (u'Wings', u'NNP')
                     (u'vada', u'VB'), (u'Pao', u'NNP'), (u'Bao', u'NNP')
                     (u'bombay', u'VB'), (u'Bachelors', u'NNP'), (u'Sandwich', u'NNP'), 
                     (u'Mile', u'NNP'), (u'High', u'NNP'), (u'Club', u'NNP'), (u'Veg', u'NNP'), (u'Sandwich', u'NNP'),
                '6444939', 
                
                '6500757', 
                
                '854440'
        
                cost:
                '4971051' 
                    (u'prices', u'NNS'), (u'are', u'VBP'), (u'very', u'RB'), (u'cheap', u'JJ')
                '3948891', 
                
                '5767031', 
                
                '6444939', 
                
                '6500757', 
                
                '854440'
                        (u'a', u'DT'), (u'hole', u'NN'), (u'on', u'IN'), (u'pockets', u'NNS')

                ambience
                '4971051' 
                    (u'place', u'NN'), (u'is', u'VBZ'), (u'creatively', u'RB'), (u'decorated', u'VBN'),
                '3948891', 
                    (u'the', u'DT'), (u'interiors', u'NNS'), (u'are', u'VBP'), (u'done', u'VBN'), (u'in', u'IN'), (u'a', u'DT'), (u'very', u'RB'), (u'interesting', u'JJ'), (u'manner', u'NN')
                '5767031', 
                    (u'interiors', u'NNS'), (u'are', u'VBP'), (u'eye', u'NN'), (u'catching', u'VBG'), (u'and', u'CC'), (u'quirky', u'JJ')
                '6444939', 
                
                '6500757', 
                
                '854440'

                service
                '4971051' 
                    (u'serving', u'VBG'), (u'was', u'VBD'), (u'delightful', u'JJ')
                '3948891', 
                
                '5767031', 
                    (u'serve', u'VBP'), (u'drinks', u'NNS'), (u'and', u'CC'), (u'food', u'NN'), (u'in', u'IN'), (u'some', u'DT'), (u'interesting', u'JJ'), (u'glasses', u'NNS')

                '6444939', 
                
                '6500757', 
                
                '854440'
                
                overall
                '3948891', 
                    (u'the', u'DT'), (u'place', u'NN'), (u'is', u'VBZ'), (u'huge', u'JJ') 
                '5767031', 
                    (u'brimming', u'VBG'), (u'with', u'IN'), (u'people', u'NNS'),
                '6444939', 
                
                '6500757', 
                
                '854440'


"""
##TODO: Make sure that while shifting on new servers, a script has to be wriiten to install java and stanforn pos tagger files
##http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html



import os
import sys
import inspect
import nltk
import re
from functools import wraps
from nltk.tag.hunpos import HunposTagger
from textblob.np_extractors import ConllExtractor
from textblob import TextBlob
from nltk.tag.stanford import POSTagger
from topia.termextract import extract  
db_script_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, db_script_path)
#from get_reviews import GetReview
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(directory))
from MainAlgorithms import InMemoryMainClassifier, timeit, cd, path_parent_dir, path_trainers_file, path_in_memory_classifiers
stanford_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(stanford_file_path))


def need_pos_tagged(pos_tagged):  
        def tags_decorator(func):
                @wraps(func)
                def func_wrapper(self, *args, **kwargs):
                        if pos_tagged and type(self.list_of_sentences[0]) != list :
                                raise StandardError("The pos tagger you are trying run needs pos tagged list of sentences\
                                        Please try some other pos tagger which doesnt require word tokenized sentences")
                        func(self, *args, **kwargs)
                return func_wrapper 
        return tags_decorator  



class NounPhrases:
        def __init__(self, list_of_sentences, default_np_extractor=None, regexp_grammer=None, if_postagged=False):
		"""
                Args:
                        list_of_sentences: A list of lists with each element is a list of sentences which is pos tagged
                        Example:
                                [[('I', 'PRP'), ('went', 'VBD'), ('there', 'RB'), ('for', 'IN'), ('phirni', 'NN')], [], [], ...]

                        default_np_extractor:
                                    if a list been passed then the noun phrases from various np_extractors will be appended
                                    if a string is passed, only the noun phrases from that np extractor will be appended
                                    Options
                                        regex_np_extractor
                                        regex_textblob_conll_np
                                        textblob_np_conll
                                        textblob_np_base

                """

                self.if_postagged = if_postagged
                self.noun_phrases = list()
                self.conll_extractor = ConllExtractor()
                self.topia_extractor = extract.TermExtractor()

                self.list_of_sentences = list_of_sentences
                self.np_extractor = ("textblob_np_conll", default_np_extractor)[default_np_extractor != None]
                if not regexp_grammer:
                        self.regexp_grammer = r"CustomNounP:{<JJ|VB|FW|VBN>?<NN.*>*<NN.*>}"

                eval("self.{0}()".format(self.np_extractor)) 
               
                self.noun_phrases = {self.np_extractor: self.noun_phrases}
                
                return 

        @need_pos_tagged(True)
        def regex_np_extractor(self):
                """
                We need convert_to_tuple method because mongodb doesnt save tuple and converts into a list
                so when we pick dat from mongodb it gives data of this type
                __text = [[[u'this', u'DT'], [u'is', u'VBZ'], [u'one', u'CD'], [u'of', u'IN'], [u'the', u'DT'], [u'good', u'JJ']]

                if we pass this text to __parser it gives chunk error, 
                so to convert this sentnece into the form 
                __text = [[(u'this', u'DT'), (u'is', u'VBZ'), (u'one', u'CD'), (u'of', u'IN'), (u'the', u'DT'), (u'good', u'JJ')]
                we need convert_to_tuple method

                """
                def convert_to_tuple(element):
                        return tuple(element)
            
                __parser = nltk.RegexpParser(self.regexp_grammer)
                for __sentence in self.list_of_sentences:
                        print "This is the sentence that got into noun phrase algorithm %s"%__sentence
                        __sentence = map(convert_to_tuple, __sentence)
                        tree = __parser.parse(__sentence)
                        for subtree in tree.subtrees(filter = lambda t: t.label()=='CustomNounP'):
                                self.noun_phrases.append(" ".join([e[0] for e in subtree.leaves()]))
                return



        @need_pos_tagged(False)
	def textblob_np_conll(self):
                for __sentence in self.list_of_sentences:
		        __sentence = " ".join([element[0] for element in __sentence])
                        blob = TextBlob(__sentence, np_extractor=self.conll_extractor)
                        self.noun_phrases.append(list(blob.noun_phrases))
                return
        
        @need_pos_tagged(False)
        def textblob_np_base(self):
                for __sentence in self.list_of_sentences:
		        blob = TextBlob(__sentence)
		        self.noun_phrases.append(blob.noun_phrases)
                return


        @need_pos_tagged(True)
        def regex_textblob_conll_np(self):
                """
                Gives a union of the noun phrases of regex grammer and text blob conll noun phrases
                """
                __parser = nltk.RegexpParser(self.regexp_grammer)
                for __sentence in self.list_of_sentences:
                        __noun_phrases = list()
		        blob = TextBlob(" ".join([_word[0] for _word in __sentence]),)
                        tree = __parser.parse(__sentence)
                        for subtree in tree.subtrees(filter = lambda t: t.label()=='CustomNounP'):
                                __noun_phrases.append(" ".join([e[0] for e in subtree.leaves()]))
                        __union = list(set(__noun_phrases)|set(blob.noun_phrases))
                        self.noun_phrases.append(__union)
                return

        @need_pos_tagged(False)
        def topia_n_textblob(self):
                """
                if_postagged:
                        Default: False
                        if false, that means a list of sentences who are not postagged being provided to
                        this method
                """

                if self.if_postagged:
		        self.list_of_sentences = [" ".join([_word[0] for _word in __sentence]) for __sentence in self.list_of_sentences]
                

                for __sentence in self.list_of_sentences:
                        blob = TextBlob(__sentence, np_extractor=self.conll_extractor)
                        nouns = self.topia_extractor(__sentence)
                        #print list(set.union(set(blob.noun_phrases), set([e[0] for e in nouns])))
                        self.noun_phrases.append(list(set.union(set([np.lower() for np in blob.noun_phrases]), set([e[0].lower() for e in nouns])))) 

        
        @need_pos_tagged(False)
        def topia(self):
                """
                if_postagged:
                        Default: False
                        if false, that means a list of sentences who are not postagged being provided to
                        this method
                """

                if self.if_postagged:
		        self.list_of_sentences = [" ".join([_word[0] for _word in __sentence]) for __sentence in self.list_of_sentences]
                

                for __sentence in self.list_of_sentences:
                        nouns = self.topia_extractor(__sentence)
                        #print list(set.union(set(blob.noun_phrases), set([e[0] for e in nouns])))
                        self.noun_phrases.append([e[0].lower() for e in nouns]) 



if __name__ == "__main__":
        text = [ [(u'i', u'LS'), (u'wanted', u'VBD'), (u'to', u'TO'), (u'go', u'VB'), (u'for', u'IN'), (u'teppanyaki', u'JJ'), (u'grill', u'NN'), (u'since', u'IN'), (u'i', u'FW'), (u'never', u'RB'), (u'tried', u'VBD'), (u'it', u'PRP'), (u'in', u'IN'), (u'Delhi', u'NNP'), (u'(', u'FW'), (u'i', u'FW'), (u'had', u'VBD'), (u'it', u'PRP'), (u'last', u'JJ'), (u'...', u':')], [(u'we', u'PRP'), (u'had', u'VBD'), (u'a', u'DT'), (u'portion', u'NN'), (u'of', u'IN'), (u'both', u'CC'), (u'the', u'DT'), (u'dishes', u'NNS'), (u'and', u'CC'), (u'called', u'VBD'), (u'up', u'RP'), (u'server', u'NN'), (u'again', u'RB'), (u'with', u'IN'), (u'menu', u'NN'), (u'to', u'TO'), (u'confirm', u'VB'), (u'the', u'DT'), (u'ingredients', u'NNS'), (u'and', u'CC'), (u'asked', u'VBD'), (u'him', u'PRP'), (u'to', u'TO'), (u'match', u'VB'), (u'the', u'DT'), (u'dish', u'NN'), (u'with', u'IN'), (u'contents', u'NNS'), (u'mentioned', u'VBN'), (u'in', u'IN'), (u'menu', u'NN'), (u'.', u'.')]]
        
        new_text = [[(u'the', u'DT'), (u'chocolate', u'NN'), (u'chip', u'NN'), (u'cookie', u'NN'), (u'isa', u'NN'), (u'good', u'JJ'), (u'side', u'NN'), (u'item', u'NN'), (u'.', u'.')]]

        __text = [[[u'this', u'DT'], [u'is', u'VBZ'], [u'one', u'CD'], [u'of', u'IN'], [u'the', u'DT'], [u'good', u'JJ'], [u'subway', u'NN'], [u'joints', u'NNS'], [u'and', u'CC'], [u'one', u'CD'], [u'that', u'WDT'], [u'has', u'VBZ'], [u'stayed', u'VBN'], [u'for', u'IN'], [u'a', u'DT'], [u'good', u'JJ'], [u'long', u'JJ'], [u'period', u'NN'], [u'of', u'IN'], [u'time', u'NN'], [u'.', u'.']]]


        __text =  ['try their Paneer Chilli Pepper starter. Pizzas and risotto too was good.3. Drinks - here is an interesting (read weird) fact..even through they have numerous drinks in the menu, on a Friday night (when I visited the place) they were serving only specific brands of liquor.']
        #instance = NounPhrases(new_text, default_np_extractor="textblob_np_conll")
        instance = NounPhrases(__text, default_np_extractor="topia_n_textblob")
        __l = instance.noun_phrases
        print __l

