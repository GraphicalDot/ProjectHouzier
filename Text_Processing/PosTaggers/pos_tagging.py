#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 3 february, 2015
For the pos tagging of the list of sentences
"""
import os
import sys
import subprocess
import warnings
from itertools import chain
from textblob import TextBlob      
from functools import wraps
from nltk import wordpunct_tokenize
from nltk import pos_tag as nltk_pos_tag
from nltk.tag.hunpos import HunposTagger
from nltk.tag.stanford import POSTagger
stanford_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))                                    
sys.path.append(os.path.join(stanford_file_path))  

dir_name = os.path.dirname(os.path.abspath(__file__))                           
print '{0}/hunpos-1.0-linux/en_wsj.model'.format(dir_name)
def need_word_tokenization(word_tokenize):
        def tags_decorator(func):
                @wraps(func)
                def func_wrapper(self, *args, **kwargs):
                        if word_tokenize and type(self.list_of_sentences[0]) != list : 
                                raise StandardError("This Pos tagger needs a Word tokenized list of sentences, Please try some other pos tagger\
                                        which doesnt require word tokenized sentences")
                        func(self, *args, **kwargs)
                return func_wrapper
        return tags_decorator






class PosTaggers:
        #os.environ["JAVA_HOME"] = "{0}/ForStanford/jdk1.8.0_31/jre/bin/".format(stanford_file_path)
        #stanford_jar_file = "{0}/ForStanford/stanford-postagger.jar".format(stanford_file_path) 
        #stanford_tagger = "{0}/ForStanford/models/english-bidirectional-distsim.tagger".format(stanford_file_path) 
        def __init__(self, list_of_sentences, default_pos_tagger=None, list_of_sentences_type=None):
                """
                Args:
                    list_of_sentences:
                        list of lists with each element in the main list as the sentence which is word_tokenized

                        if you want to pos tag on multithread just pass the word tokenized sentence in a list, 
                        that obviously will be considered as a list of length a length one
                
                    default_pos_tagger:
                        type: string
                        options:
                                stan_pos_tagger
                                hunpos_pos_tagger
                                nltk_pos_tagger
                                textblob_pos_tagger

                
                need_word_tokenization decorator two things
                    if the pos tagger needs a word tokenized list of sentences or list of sentences
                    @need_word_tokenization(True) means that this pos tagger needs word tokenized list 
                    and then checks whether self.list_of_sentences is that or not 
                """

                self.check_if_hunpos() 
                self.hunpos_tagger = HunposTagger('{0}/hunpos-1.0-linux/en_wsj.model'.format(dir_name),
                                                    '{0}/hunpos-1.0-linux/hunpos-tag'.format(dir_name))
                #self.stanford_tagger = POSTagger(self.stanford_tagger, self.stanford_jar_file) 
                
                self.list_of_sentences = list_of_sentences
                self.pos_tagged_sentences = list()
                self.pos_tagger = ("nltk_pos_tagger", default_pos_tagger)[default_pos_tagger != None]                
                eval("self.{0}()".format(self.pos_tagger))

                self.pos_tagged_sentences = {self.pos_tagger: self.pos_tagged_sentences}
                return 

        def check_if_hunpos(self):
                """
                This method checks if the executabled of hunpos exists or not
                """
                if not os.path.exists("{0}/hunpos-1.0-linux".format(dir_name)):
                        warnings.warn("Downloading the hun pos tagger files as they werent here,to be used for tagging")
                        subprocess.call(["wget", "https://hunpos.googlecode.com/files/hunpos-1.0-linux.tgz"])
                        subprocess.call(["wget", "https://hunpos.googlecode.com/files/en_wsj.model.gz"])
                        subprocess.call(["tar", "xvfz", "hunpos-1.0-linux.tgz"])
                        subprocess.call(["gunzip", "en_wsj.model.gz"])
                        subprocess.call(["mv", "en_wsj.model", "hunpos-1.0-linux"])
                        subprocess.call(["rm", "-rf", "en_wsj.model.gz.1"])
                        subprocess.call(["rm", "-rf", "hunpos-1.0-linux.tgz"]) 


        


        @need_word_tokenization(True)
        def hunpos_pos_tagger(self):
                for __sentence in self.list_of_sentences:
                        self.pos_tagged_sentences.append(self.hunpos_tagger.tag(__sentence))

                return

        @need_word_tokenization(True)
        def stan_pos_tagger(self):
                for __sentence in self.list_of_sentences:
                        try:
                            __tagged_sentence = self.stanford_tagger.tag(__sentence)
                            self.pos_tagged_sentences.append(__tagged_sentence)
                        except Exception:
                            pass
                return

        @need_word_tokenization(False)
        def textblob_pos_tagger(self):
                for __sentence in self.list_of_sentences:
                        blob = TextBlob(__sentence)
                        self.pos_tagged_sentences.append(blob.pos_tags)
                return 

        @need_word_tokenization(True)
        def nltk_pos_tagger(self):
                for __sentence in self.list_of_sentences:
                        self.pos_tagged_sentences.append(nltk_pos_tag(__sentence))
                return


if __name__ =="__main__":
        text = [[u'i', u'like', u'how', u'we', u'young', u'people', u'are', u'realizing', u'the', u'importance', u'of', u'eating', u'healthy', u'rather', u'than', u'eating', u'junk', u'.']]


        p = PosTaggers(text, default_pos_tagger="hunpos_pos_tagger")
        print p.pos_tagged_sentences
