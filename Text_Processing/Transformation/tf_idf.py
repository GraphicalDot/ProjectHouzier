
#!/usr/bin/env python
from sklearn.feature_extraction.text import TfidfTransformer



class HouzierTfIdf(object):
            """
            The output from Vectorization return term frequency 
            Term-frequency to represent textual information in the vector space.
            However, the main problem with the term-frequency approach is that
            it scales up frequent terms and scales down rare terms which are
            empirically more informative than the high frequency terms. The
            basic intuition is that a term that occurs frequently in many
            documents is not a good discriminator, and really makes sense (at
            least in many experimental tests); the important question here is:
            why would you, in a classification problem for instance, emphasize
            a term which is almost present in the entire corpus of your
            documents ?

             tf(t,d) which is actually the term count of the term t in
             the document d. The use of this simple term frequency could lead
             us to problems like keyword spamming, which is when we have
             a repeated term in a document with the purpose of improving its
             ranking on an IR (Information Retrieval) system or even create
             a bias towards long documents, making them look more important
             than they are just because of the high frequency of the term in
             the document.


            """
            
            def __init__(self):
                    pass



            @staticmethod
            def sklean_tf_idf(term_frequency_matrix, if_dense_matrix):
                    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
                    tfidf_matrix = tfidf.fit(term_frequency_matrix)
                    if if_dense_matrix:
                            tfidf_matrix = tfidf_matrix.todense()

                    return tfidf_matrix



