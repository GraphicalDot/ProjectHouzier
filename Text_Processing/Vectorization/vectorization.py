#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import os  # for os.path.basename
#import matplotlib.pyplot as plt
#If you want ot import on macosx then use frameworkpython function mentioned in
#bashrc file

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import operator
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import PolynomialFeatures
from os.path import dirname, abspath 

filename = dirname(dirname(abspath(__file__)))
sys.path.append(filename)
print filename 
import PreProcessingText

sys.path.append(dirname(filename))
from configs import cd



from sklearn.externals import joblib
#https://dandelion.eu/semantic-text/entity-extraction-demo/
#http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/


class HouzierVectorizer(object):

        def __init__(self, sentences, path, file_name_vectorizer, use_dense_matrix=False, enable_print=False): 
                
            
                """
                Args:
                    sentences: training_sentences 
                    preprocessor_callable: a function which will preprocess the
                            text before fetching into the vectorizer, as in to
                            remove hmtl characters etc

                    use_dense_matrix: By default CountVectorizer spits out
                            a sparse matrix, To convert it into dense matrix
                            which is required by sevral agorithms use True

                    enable_print: If you want to print the features count and 
                            features name while measurin performance use True.

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
                document term matrix
                """
                self.sentences = sentences
                self.path = path
                self.file_name_vectorizer = file_name_vectorizer
                self.use_dense_matrix = use_dense_matrix
                self.enable_print = enable_print
                self.stemmer = SnowballStemmer("english")
                self.ngram_range = [1,  3]
                #self.preprocessor_callable = preprocessor_callable

                return 




        def count_vectorize(self): 
                """
                token_pattern=u'(?u)\\b\\w\\w+\\b' removes single word from the
                vocabullary
                """
                
                
                
                #vectorizer = CountVectorizer(preprocessor=preprocess, analyzer=stemmed_words, ngram_range=(2, 6))
                vectorizer = CountVectorizer(ngram_range=(1, 3))
                
                dtm = vectorizer.fit_transform(self.sentences)  # a sparse
                #this is a sparse matrix to convert it into dense matrix
                #use    dt.todense()
                with cd(self.path):
                        joblib.dump(vectorizer.vocabulary_, self.file_name_vectorizer)
                if self.enable_print:
                        print sorted(vectorizer.vocabulary_.items(),
                             key=operator.itemgetter(1))
                
                print "shape of the document matrix is rows=%s,columns=%s"%dtm.shape
                if self.use_dense_matrix:
                        self.dtm = dtm.todense()
                else:
                        self.dtm = dtm

                return self.dtm

        def return_vectorizer(self):
                with cd(self.path):
                        vocabulary = joblib.load(self.file_name_vectorizer)
                return vocabulary

                


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


"""
if __name__ == "__main__":
        f = open("sentiment_training_sentences.pickle", "rb")
        sentiment_sentences = pickle.load(f)
        cls = HouzierVectorizer(sentiment_sentences[0:10],
                                PreProcessingText.PreProcessText)
        cls.count_vectorize()
"""
                            
