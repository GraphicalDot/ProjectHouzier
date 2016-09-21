
#!/usr/bin/env python


from PreProcessingText import PreProcessText
from Vectorization import HouzierVectorizer
from Transformation import  HouzierTfIdf
from TrainingData.MongoData import TrainingMongoData
from nltk.stem import SnowballStemmer

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer





class SentimentClassifiers(object): 

        @staticmethod
        def snowball_stemmer(sentences):
                stemmer = SnowballStemmer("english")
                return [stemmer.stem(sent) for sent in sentences]


        @staticmethod 
        def pre_process_text(sentences):
                return [PreProcessText.process(sent) for sent in sentences]


        @staticmethod
        def svm():
                sentiments, sentences = zip(*TrainingMongoData.sentiment_data_three_categories())
                sentences = SentimentClassifiers.snowball_stemmer(sentences)
                sentences = SentimentClassifiers.pre_process_text(sentences)
                #sentences = [PreProcessText.process(sent) for sent in sentences] 
                vectorize_class = HouzierVectorizer(sentences, False, True)
                x_vectorize = vectorize_class.count_vectorize()


                tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
                
                x_transform = tfidf.fit_transform(x_vectorize)
                """


                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
                X_train = vectorizer.fit_transform(sentences)
                """

                # This dataset is way too high-dimensional. Better do PCA:
                pca = PCA(n_components=2)
                #
                # # Maybe some original features where good, too?
                selection = SelectKBest(k=1)
                #
                combined_features = FeatureUnion([("pca", pca), ("univ_select",
                                                                 selection)])

                X_features = combined_features.fit_transform(x_transform.toarray(),
                                                             sentiments)
                
                return 
         



class SubCategoryClassifiers(object):
        pass




if __name__ == "__main__":
        SentimentClassifiers.svm()









