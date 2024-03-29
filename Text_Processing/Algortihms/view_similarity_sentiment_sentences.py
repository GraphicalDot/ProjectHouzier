#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import os  # for os.path.basename
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import operator
from nltk.stem import SnowballStemmer




filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(filename)
import PreProcessingText



#f = open("sentiment_training_sentences.pickle", "rb")
#sentiment_sentences = pickle.load(f)

class HouzierVectorizer(object):

        def __init__(self, sentiment_sentences_list, ngram_range=[1, 6],
                     preprocessor_callable):
                """
                tags = [
                  "python, tools",
                    "linux, tools, ubuntu",
                      "distributed systems, linux, networking, tools",
                      ]
                      The next step is:

                      from sklearn.feature_extraction.text import
                      CountVectorizer
                      vec = CountVectorizer(tokenizer=tokenize)
                      data = vec.fit_transform(tags).toarray()
                      print data
                      Where we get

                      [[0 0 0 1 1 0]
                       [0 1 0 0 1 1]
                        [1 1 1 0 1 0]]


                vectorizer = CountVectorizer(input='filename')
                dtm = vectorizer.fit_transform(filenames)  # a sparse
                vocab = vectorizer.get_feature_names() 
                #Get feature index
                list(vocab).index(word)

                CountVectorizer returns a sparse matrix by default, consider
                a 4000 by 50000 matrix of word frequencies that is 60% zeros.

                dtm = dtm.toarray()  # convert to a regular array
                vocab = np.array(vocab)

                dtm.shape (15581, 9362)
                So here we have 15581 rows corresponding to 15581
                training_sentences and 9362 is a vocabulary.This is called
                document term matri
                """

                self.training_sentiment_tags, self.training_sentences = zip(*sentiment_sentences_list)
                self.stemmer = SnowballStemmer("english")
                self.ngram_range = ngram_range
                self.preprocessor_callable = preprocessor_callable

                return 




        def count_vectorize(self): 
                analyzer = CountVectorizer().build_analyzer()
                preprocessor = CountVectorizer().build_preprocessor()
                
                
                def stemmed_words(doc):
                        return (self.stemmer.stem(w) for w in analyzer(doc))
                

                def preprocess(docc):
                        return (self.preprocessor_callable(w) for w in analyzer(doc))

                
                
                
                vectorizer = CountVectorizer(preprocessor=preprocess, analyzer=stemmed_words, ngram_range=(1, 6))
                
                dtm = vectorizer.fit_transform(self.training_sentences)  # a sparse
                print "These are some features"
                print sorted(vectorizer.vocabulary_.items(),
                             key=operator.itemgetter(1))
                
                print len(vectorizer.vocabulary_)
                #print vectorizer.get_feature_names()
                print "shape of the document matrix is rows=%s,columns=%s"%dtm.shape
                self.dtm = dtm
                return


        def _cosine_similarity(self):

                ##This below statement calculates the cosine similarity between
                ##rows, so dist must have a shape of (15581, 15581), 
                #So it basically calulates the cosine similairty with a sentece
                ##with all the remaining sentences.

                """
                                   Austen_Emma Austen_Pride Austen_Sense CBronte_Jane CBronte_Professor CBronte_Villette
                        Austen_Emma        -0.00    0.02           0.03       0.05            0.06          0.05
                        Austen_Pride        0.02    0.00           0.02       0.05            0.04          0.04
                        Austen_Sense        0.03    0.02           0.00       0.06            0.05          0.05
                        CBronte_Jane        0.05    0.05           0.06       0.00            0.02          0.01
                        CBronte_Professor   0.06    0.04           0.05       0.02           -0.00          0.01
                        CBronte_Villette    0.05    0.04           0.05       0.01            0.01         -0.00
            
                Visualizing distances
                """
                dist = 1 - cosine_similarity(self.dtm)
                np.round(dist, 2)
                mds = MDS(n_components=2, dissimilarity="precomputed",
                       random_state=1)

                pos = mds.fit_transform(dist)  # shape (n_components,n_samples)

                xs, ys = pos[:, 0], pos[:, 1]
                for x, y, tag in zip(xs, ys, self.training_sentiment_tags):
                        _tag = tag.split("-")[-1]
                        color = 'red' if _tag == "negative" else 'black'
                        plt.scatter(x, y, c=color)
                        plt.text(x, y, _tag[0:3])
                plt.show()
                return 



if __name__ == "__main__":
        cls = SentimentSimilaritySenteceView()
        cls.run()

                            
