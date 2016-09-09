#!/usr/bin/env pyhon
#-*- coding: utf-8 -*-



class NpClustering:
        def __init__(self, list_of_noun_phrases, default_cluster_algorithm=None):
                """
                Args:
                        list_of_noun_phrases :
                                list of lists with each list of the form
                                [noun_phrase, frequency, sentiment]

                        default_cluster_algorithm: k-means
                """
                self.noun_phrases_cluster = list()
                self.list_of_noun_phrases = list_of_noun_phrases
                self.ner = ("stanford_ner", default_cluster_algorithm)[default_cluster_algorithm != None]
                eval("self.{0}()".format(self.ner))
                self.ners = {self.ner: self.ners}
                return


        def k_means(self):
                return

                  
