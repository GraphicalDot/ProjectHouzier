#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 4 February, 2015
This is the file which deals with the word tokenization of the sentences

"""
from nltk import word_tokenize
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.tokenize import TreebankWordTokenizer

class WordTokenize:
        def __init__(self, list_of_sentences, default_word_tokenizer=None):
            """
            Args:
                list_of_sentences: A list of sentences 
            
            default_word_tokenizer:
                    options
                        punkt_n_treebank
                        punkt_tokenize
                        treebank_tokenize
            
            """
            self.word_tokenized_list = list()
            self.list_of_sentences = list_of_sentences
            self.word_tokenize = ("punkt_n_treebank", default_word_tokenizer)[default_word_tokenizer != None] 
            eval("self.{0}()".format(self.word_tokenize))

            self.word_tokenized_list = {self.word_tokenize: self.word_tokenized_list}
            return
    

        def to_unicode_or_bust(self, obj, encoding='utf-8'):
                if isinstance(obj, basestring): 
                        if not isinstance(obj, unicode):
                                obj = unicode(obj, encoding,)
                return obj 

        def punkt_n_treebank(self):
                for __sentence in self.list_of_sentences:
                        __sentence = self.to_unicode_or_bust(__sentence)
                        self.word_tokenized_list.append(word_tokenize(__sentence))
                return


        def punkt_tokenize(self):
                for __sentence in self.list_of_sentences:
                        __sentence = self.to_unicode_or_bust(__sentence)
                        self.word_tokenized_list.append(PunktWordTokenizer().tokenize(__sentence))
                return

        def treebank_tokenize(self):
                for __sentence in self.list_of_sentences:
                        __sentence = self.to_unicode_or_bust(__sentence)
                        self.word_tokenized_list.append(TreebankWordTokenizer().tokenize(__sentence))
                return
                
"""
if __name__ == "__main__":
        p = WordTokenize(["I went there to have chicken pizza"], default_word_tokenizer="treebank_tokenize")
        print p.word_tokenized_list
"""

