
#!/usr/bin/pypy

from os.path import dirname, abspath
from PreProcessingText import PreProcessText
from Vectorization import HouzierVectorizer
from Transformation import  HouzierTfIdf
from TrainingData.MongoData import TrainingMongoData
from nltk.stem import SnowballStemmer
from configs import base_dir, cd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC 
from sklearn.externals import joblib
from  CoreNLPScripts import  CoreNLPScripts
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
#on mac print option +# to search throught he file
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

parent = dirname(abspath(__file__))

def analyse_sentences(result):
        part_of_speech_list = list()
        for element in result["sentences"][0]["words"]:
                part_of_speech_list.append((element[0], element[1]["PartOfSpeech"]))
                
        sentence = list()
        for (word, pos) in part_of_speech_list:
                if pos in ["DT", "NNS", "NNP", "CC"]:
                        word = "DUMMY"
                sentence.append(word)
		

        print " ".join(sentence)
                    
        return " ".join(sentence) 



def transform(sentence):
		for rel, _, head, word, n in sentence['dependencies']:
			n = int(n)
			word_info = sentence['words'][n - 1][1]
			tag = word_info['PartOfSpeech']
			lemma = word_info['Lemma']
			if rel == 'root':
					#NLTK expects that the root relation is labelled as ROOT!
					rel = 'ROOT'
			# Hack: Return values we don't know as '_'.
			# Also, consider tag and ctag to be equal.
			# n is used to sort words as they appear in the sentence.
			yield n, '_', word, lemma, tag, tag, '_', head, rel, '_', '_'

class SentimentClassifiers(object): 

        @staticmethod
        def snowball_stemmer(sentences):
                stemmer = SnowballStemmer("english")
                return [stemmer.stem(sent) for sent in sentences]


        @staticmethod 
        def pre_process_text(sentences):
                return [PreProcessText.process(sent) for sent in sentences]


        @staticmethod
        def svm_bagclassifier(sentiment_data, file_name_classifier, file_name_vectorizer):
                """
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
                X_train = vectorizer.fit_transform(sentences)
                """
                sentiments, sentences=zip(*sentiment_data)
                sentences = SentimentClassifiers.snowball_stemmer(sentences)
                sentences = SentimentClassifiers.pre_process_text(sentences)
                vectorize_class = HouzierVectorizer(sentences,
                                                    file_name_vectorizer, False, False)
                
                
                ##getting features list
                x_vectorize = vectorize_class.count_vectorize()
                tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)

                ##convert them into term frequency
                x_transform = tfidf.fit_transform(x_vectorize)


                print "Feature after vectorization of the data [%s, %s]"%x_transform.shape
                ##Going for feature selection
                # This dataset is way too high-dimensional. Better do PCA:
                pca = PCA()
                #
                ## Maybe some original features where good, too?
                ##this will select features basec on chi2 test 
                selection = SelectKBest(chi2, k=15)
                combined_features = FeatureUnion([("pca", pca), ("univ_select",
                                                                 selection)])


                X_features = combined_features.fit_transform(x_transform.toarray(),
                                                             sentiments)


                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape

               
                #http://stackoverflow.com/questions/32934267/feature-union-of-hetereogenous-features

                #clf = SVC(C=1, kernel="linear", gamma=.001, probability=True, class_weight='auto')
                
                n_estimators = 20
                classifier = BaggingClassifier(SVC(kernel='linear',
                                                            gamma=.001, 
                                                            class_weight="balanced"),
                                                        max_samples=1.0/n_estimators,
                                                        n_estimators=n_estimators,
                                                                    n_jobs=-1, 
                                                                   bootstrap=False)
                
                import numpy as np 
                classifier.fit(X_features, sentiments)

                print classifier.classes_
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(classifier, file_name_classifier)

                print "Storing Classifier with joblib"
                ##example to build your own vectorizer 
                ##http://stackoverflow.com/questions/31744519/load-pickled-classifier-data-vocabulary-not-fitted-error
                from sklearn.feature_extraction.text import CountVectorizer
                #count_vectorizer = CountVectorizer()
                examples = ['Free Viagra call today!', "I am dissapointed i \
                            you", "i am not good", "I'm going to attend theLinux users group tomorrow."]
                #example_counts= example_counts.toarray()
                vocabulary_to_load = vectorize_class.return_vectorizer()
                #vectorize_class = HouzierVectorizer(examples, True, False)
                #x_vectorize = vectorize_class.count_vectorize()
                
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary_to_load) 
                example_counts = loaded_vectorizer.transform(examples)

                
                print example_counts, example_counts.shape 

                f = combined_features.transform(example_counts.toarray())
                print f.shape

                print dir(f)
                #predictions = classifier.predict(f)
                predictions = classifier.predict(f)
                for sent, tag in zip(examples, predictions):
                                     print sent, tag
                return 
         

        @staticmethod
        def print_report():
                expected = y
                predicted = model.predict(X)
                # summarize the fit of the model
                print(metrics.classification_report(expected, predicted))
                print(metrics.confusion_matrix(expected, predicted))
                return 


        @staticmethod
        def svm(sentiment_data, file_name_classifier, file_name_vectorizer):
                """
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
                X_train = vectorizer.fit_transform(sentences)
                """
                sentiments, sentences=zip(*sentiment_data[0:3000])
                sentences = SentimentClassifiers.snowball_stemmer(sentences)
                sentences = SentimentClassifiers.pre_process_text(sentences)
                vectorize_class = HouzierVectorizer(sentences,
                                                    file_name_vectorizer, False, False)
                
                
                ##getting features list
                x_vectorize = vectorize_class.count_vectorize()
                tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)

                ##convert them into term frequency
                x_transform = tfidf.fit_transform(x_vectorize)


                print "Feature after vectorization of the data [%s, %s]"%x_transform.shape
                ##Going for feature selection
                # This dataset is way too high-dimensional. Better do PCA:
                pca = PCA(whiten=True)
                #
                ## Maybe some original features where good, too?
                ##this will select features basec on chi2 test 
                selection = SelectKBest(chi2, k="all")
                combined_features = FeatureUnion([("pca", pca), ("univ_select",
                                                                 selection)])


                X_features = combined_features.fit_transform(x_transform.toarray(),
                                                             sentiments)


                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape

               
                #http://stackoverflow.com/questions/32934267/feature-union-of-hetereogenous-features

                classifier = SVC(C=1, kernel="linear", gamma=.0001)
                classifier.fit(X_features, sentiments)
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(classifier, file_name_classifier)

                ##example to build your own vectorizer 
                ##http://stackoverflow.com/questions/31744519/load-pickled-classifier-data-vocabulary-not-fitted-error
                from sklearn.feature_extraction.text import CountVectorizer
                #count_vectorizer = CountVectorizer()
                examples = ['Free Viagra call today!', "I am dissapointed i \
                            you", "i am not good", "I'm going to attend theLinux users group tomorrow."]
                #example_counts= example_counts.toarray()
                vocabulary_to_load = vectorize_class.return_vectorizer()
                #vectorize_class = HouzierVectorizer(examples, True, False)
                #x_vectorize = vectorize_class.count_vectorize()
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary_to_load) 
                example_counts = loaded_vectorizer.transform(examples)

                
                print example_counts, example_counts.shape 

                f = combined_features.transform(example_counts.toarray())
                print f.shape

            
                predictions = classifier.predict(f)
                for sent, tag in zip(examples, predictions):
                                     print sent, tag
                return 
         



class SubCategoryClassifiers(object):
        pass




if __name__ == "__main__":
    SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_three_categories(),
                                     "svm_linear_kernel",
                                     "linear_kernel_vectorizer.pkl")
    SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_after_corenlp_analysis(),
                                     "svm_linear_kernel_corenlp_data",
                                     "linear_kernel_vetorizer_corenlp.pkl")









