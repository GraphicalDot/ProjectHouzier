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
grammer = r"""CustomNounP:{<JJ|VB|FW>?<NN.*>*<NN.*>}"""
grammer = r"""CustomNounP:{<JJ|VB|FW|VBN>?<NN.*>*<NN.*>}"""


"""
##TODO: Make sure that while shifting on new servers, a script has to be wriiten to install java and stanforn pos tagger files
##http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

from Blob import ProcessingWithBlobInMemory
import os
import sys
import nltk
import inspect
from textblob import TextBlob
import re
from nltk.tag.stanford import POSTagger

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stanford_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(os.path.join(directory))
sys.path.append(os.path.join(stanford_file_path))


from MainAlgorithms import InMemoryMainClassifier, timeit, cd, path_parent_dir, path_trainers_file, path_in_memory_classifiers
from Sentence_Tokenization import SentenceTokenizationOnRegexOnInterjections

class TestNounPhrasesAlgorithms:
        os.environ["JAVA_HOME"] = "{0}/ForStanford/jdk1.8.0_31/jre/bin/".format(stanford_file_path)
        stanford_jar_file = "{0}/ForStanford/stanford-postagger.jar".format(stanford_file_path)
        stanford_tagger = "{0}/ForStanford/models/english-bidirectional-distsim.tagger".format(stanford_file_path)
        
        """
        Args:
                sentences: list of tuples with first element being a text and second element being a list of noun phrases extraction
        Test Grammer:
                for stanford pos taggers
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
        def __init__(self, sentences):
                self.sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                self.for_nouns_only_grammer = r"""CustomNounP:{<NN.*><NN>*<NN>}
                                                    CustomNounWithVBN: {<NN>?<VBN><NN>*<NN.*>}
                                                    CustomNounWithJJ: {<JJ><NN>.*<NN.*>}"""
                self.sentences = sentences


        def stanford_pos_tagger(self):
                tagger = POSTagger(self.stanford_tagger, self.stanford_jar_file)
                return tagger            

        def with_nltk(self):
                """
                returns a list of list
                with each element of the parent list as a list of noun phrases for the sentence in the original
                self.sentences
                """
                noun_phrases_list = list()
                for __sent, nouns in self.sentences:
                        __noun_phrases_list = list()
                        cp = nltk.RegexpParser(self.for_nouns_only_grammer)
                        for sentence in self.sent_tokenizer.tokenize(__sent):
                                tree = cp.parse(nltk.pos_tag(nltk.wordpunct_tokenize(sentence)))
                                for subtree in tree.subtrees(filter = lambda t: t.label()=='CustomNounP' or t.label()== 'CustomNounWithVBN' or \
                                                    t.label() == 'CustomNounWithJJ'):
                                        __noun_phrases_list.append(" ".join([e[0] for e in subtree.leaves()]))
                        noun_phrases_list.append(__noun_phrases_list)
                return noun_phrases_list


        @staticmethod
        def compare_results(__l1, __l2):
                """
                __l1 : Noun phrase that were expected
                __l2 : Noun Phrases that were being extraced by the custom algorithm

                """

                print "Noun Phrases expected {0}".format(__l1)
                print "Noun Phrases extracted {0}".format(__l2)
                result = list(set(__l1) - set(__l2))
                if bool(result):
                        print "Failed Algorithm"
                        print "Didnt succeed in capturing {0} \n".format(result)
                else:
                        print "success \n"


        def check_pos_tagging(self, default="stanford"):
                tagger = self.stanford_pos_tagger()
                #append this list for new review_ids
                 review_ids = ['4971051', '3948891', '5767031', '6444939', '6500757', '854440']
                for __id in review_ids:
                        __text = reviews.find_one({"review_id": __id}).get("review_text")
                        for sentence in self.sent_tokenizer.tokenize(__text): 
                                print tagger.tag(nltk.wordpunct_tokenize(sentence.decode("utf-8"))), "\n\n\n"
                                         


if __name__ == "__main__":
        sentences = [("Well the ferrero rocher shake, the kitkat shake and the chocolate bloodbath is a must have! Plus, \
            it doesn't hit your pocket!The ambience is pretty good, specially because it gives you the perfect view of the fort :) must try once!", \
            ["ferrero rocher shake", "kitkat shake", "chocolate bloodbath", "perfect view"]),

                    ("We ordered a lot of things… caesar chicken salad, chicken n cheese nachos, shawarma roll, veg pizza, penne alfredo, \
                            bbq hot dog, cottage cheese bbq, thai red curry, fried rice…for the drinks we had hazelnut coffee, cold coffee with \
                            ice cream and a mango mint shake. The mango mint shake was amazing and a really new taste for my bud. For the food…\
                            the best thing was the thin crust pizza among the appetizers. Chicken n cheese nachos were bland, even the dip was sweet\
                            and didn’t do justice to the dish.", 
                            ["caesar chicken salad", "chicken n cheese nachos", "shawarma roll", "veg pizza", "penne alfredo", "bbq hot dog",\
                            "cottage cheese bbq", "thai red curry", "fried rice", "hazelnut coffee", "cold coffee with ice cream", "mango mint shake"
                            , "thin crust pizza", "chicken n cheese nachos"])

                ]

        ins = TestNounPhrasesAlgorithms(sentences)
        tagger = ins.stanford_pos_tagger()
        print tagger.tag(nltk.wordpunct_tokenize("hey man!!!"))
        #for sentence, resulting_nouns in zip(sentences, ins.with_nltk()):
        #        TestNounPhrasesAlgorithms.compare_results(sentence[1], resulting_nouns)



