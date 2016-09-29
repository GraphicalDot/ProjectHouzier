
#!/usr/bin/pypy

from os.path import dirname, abspath
from PreProcessingText import PreProcessText
from Vectorization import HouzierVectorizer
from Transformation import  HouzierTfIdf
from TrainingData.MongoData import TrainingMongoData
from nltk.stem import SnowballStemmer
from configs import base_dir, cd
from sklearn.decomposition import PCA, RandomizedPCA, NMF, KernelPCA
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
import time 
import numpy as np
from sklearn.metrics import accuracy_score


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





class GeneralMethodsClassifiers(object):
        @staticmethod
        def snowball_stemmer(sentences):
                stemmer = SnowballStemmer("english")
                return [stemmer.stem(sent) for sent in sentences]


        @staticmethod 
        def pre_process_text(sentences):
                return [PreProcessText.process(sent) for sent in sentences]




class SentimentClassifiers(object): 


        @staticmethod
        def svm_bagclassifier(sentiment_data, file_name_classifier,
                              file_name_vectorizer, file_name_features):
                """
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
                X_train = vectorizer.fit_transform(sentences)
                """
                import time 
                start = time.time()
                sentiments, sentences=zip(*sentiment_data[0: 1000])
                sentences = GeneralMethodsClassifiers.snowball_stemmer(sentences)
                sentences = GeneralMethodsClassifiers.pre_process_text(sentences)
                vectorize_class = HouzierVectorizer(sentences,
                                                    "%s/CompiledModels/SentimentClassifiers"%base_dir,
                                                    file_name_vectorizer, False, False)
                
                
                ##getting features list
                x_vectorize = vectorize_class.count_vectorize()
                tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)

                ##convert them into term frequency
                x_transform = tfidf.fit_transform(x_vectorize)

                X_normalized = preprocessing.normalize(x_transform.toarray(), norm='l2')
                print "Feature after vectorization of the data [%s, %s]"%x_transform.shape
                ##Going for feature selection
                # This dataset is way too high-dimensional. Better do PCA:
                #pca = PCA()
                pca = KernelPCA(kernel="linear")
                #pca = RandomizedPCA()
                #pca = NMF()
                #
                ## Maybe some original features where good, too?
                ##this will select features basec on chi2 test 
                
                selection = SelectKBest(chi2, k=200)
                combined_features = FeatureUnion([("pca", pca), ("univ_select",
                                                                 selection)])


                X_features = combined_features.fit_transform(X_normalized,
                                                           sentiments)
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(combined_features, "combined_features_sentiment")


                #X_pca = pca.fit_transform(x_transform)


                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape

               
                #http://stackoverflow.com/questions/32934267/feature-union-of-hetereogenous-features

                #clf = SVC(C=1, kernel="linear", gamma=.001, probability=True, class_weight='auto')
                
                n_estimators = 20
                classifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', 
                                                            C= 1, 
                                                            gamma="auto", 
                                                            probability=True, 
                                                            decision_function_shape="ovr",
                                                            class_weight="balanced",
                                                                       ),
                                                        max_samples=1.0,
                                                        max_features= 1.0, 
                                                        n_jobs=-1, 
                                                        verbose=True, 
                                                        n_estimators=n_estimators,
                                                                   bootstrap=False))
                
                
                
                classifier.fit(X_features, sentiments)

                print classifier.classes_
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(classifier, file_name_classifier)

                print "Storing Classifier with joblib"
                ##example to build your own vectorizer 
                ##http://stackoverflow.com/questions/31744519/load-pickled-classifier-data-vocabulary-not-fitted-error
                from sklearn.feature_extraction.text import CountVectorizer
                #count_vectorizer = CountVectorizer()
                examples_negative = ['Free Viagra call today!', "I am dissapointed in you", "i am not good"] 
                examples_neutral = ["I dont know", "Sun rises in the east", "I'm going to attend theLinux users group tomorrow."]
                examples_positive = ["hey there, I am too good to be true", "An Awesome man", "A beautiful beautiful lady"]
                
                
                examples = examples_positive + examples_negative + examples_neutral


                #example_counts= example_counts.toarray()
                vocabulary_to_load = vectorize_class.return_vectorizer()
                #vectorize_class = HouzierVectorizer(examples, True, False)
                #x_vectorize = vectorize_class.count_vectorize()
                
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary_to_load) 
                example_counts = loaded_vectorizer.transform(examples)

                
                print example_counts, example_counts.shape 

                

                f = combined_features.transform(example_counts.toarray())

                predictions = classifier.predict(f)
                predict_probabilities = classifier.predict_proba(f)
                for sent, prob, tag in zip(examples, predict_probabilities, predictions):
                                     print sent, prob, tag
                
                
                print time.time() -start
                return 
         

        @staticmethod
        def print_report():
                expected = y
                predicted = model.predict(X)
                # summarize the fit of the model
                print(metrics.classification_report(expected, predicted))
                print(metrics.confusion_matrix(expected, predicted))
                return 





class TagClassifiers(object): 

        @staticmethod
        def svm_bagclassifier(tag_data, file_name_classifier,
                              file_name_vectorizer, file_name_features):
                """
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
                X_train = vectorizer.fit_transform(sentences)
                
                We have big dataset for tag classification, we cannot use
                PCA or any other flavour of PCA for the time being because its
                too heavy on computational resources.

                We just goign to uee selectkbest features which selectes best
                features from the features coumputed by CountVectorizer
                """

                training_data, test_data = tag_data
                print "Length of the training data %s"%len(training_data)

                start = time.time()
                tags, sentences = zip(*training_data[0: 1000])
                sentences = GeneralMethodsClassifiers.snowball_stemmer(sentences)
                sentences = GeneralMethodsClassifiers.pre_process_text(sentences)
                vectorize_class = HouzierVectorizer(sentences,
                                                    "%s/CompiledModels/TagClassifiers"%base_dir,
                                                    file_name_vectorizer, False, False)
                
                
                ##getting features list
                x_vectorize = vectorize_class.count_vectorize()
                tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)

                ##convert them into term frequency
                x_transform = tfidf.fit_transform(x_vectorize)

                X_normalized = preprocessing.normalize(x_transform.toarray(), norm='l2')
                print "Feature after vectorization of the data [%s, %s]"%x_transform.shape
                selection = SelectKBest(chi2, k=200)

                X_features = selection.fit_transform(X_normalized,
                                                           tags)
                with cd("%s/CompiledModels/TagClassifiers"%base_dir):
                        joblib.dump(selection, "combined_features_tag")


                #X_pca = pca.fit_transform(x_transform)


                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape

               
                #http://stackoverflow.com/questions/32934267/feature-union-of-hetereogenous-features

                #clf = SVC(C=1, kernel="linear", gamma=.001, probability=True, class_weight='auto')
                
                n_estimators = 20
                classifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', 
                                                            C= 1, 
                                                            gamma="auto", 
                                                            probability=True, 
                                                            decision_function_shape="ovr",
                                                            class_weight="balanced",
                                                                       ),
                                                        max_samples=1.0,
                                                        max_features= 1.0, 
                                                        n_jobs=-1, 
                                                        verbose=True, 
                                                        n_estimators=n_estimators,
                                                                   bootstrap=False))
                
                
                
                classifier.fit(X_features, tags)

                print classifier.classes_
                with cd("%s/CompiledModels/TagClassifiers"%base_dir):
                        joblib.dump(classifier, file_name_classifier)

                print "Storing Classifier with joblib"
                ##example to build your own vectorizer 
                ##http://stackoverflow.com/questions/31744519/load-pickled-classifier-data-vocabulary-not-fitted-error
                from sklearn.feature_extraction.text import CountVectorizer
                


                #example_counts= example_counts.toarray()
                vocabulary_to_load = vectorize_class.return_vectorizer()
                #vectorize_class = HouzierVectorizer(examples, True, False)
                #x_vectorize = vectorize_class.count_vectorize()
                
                examples, examples_target = zip(*test_data)
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary_to_load) 
                example_counts = loaded_vectorizer.transform(examples)

                
                print example_counts, example_counts.shape 

                #f = combined_features.transform(example_counts.toarray())
                f = selection.transform(example_counts.toarray())

                predictions = classifier.predict(f)
                predict_probabilities = classifier.predict_proba(f)

                print accuracy_score(examples_target, predictions)
                
                print time.time() -start
                return 
         

class SubCategoryClassifiers(object):
        pass




if __name__ == "__main__":
        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_three_categories(),
                                     "svmlk_sentiment_classifier.pkl",
                                     "lk_vectorizer_sentiment.pkl", 
                                        "sentiment_features.pkl")
        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_after_corenlp_analysis(),
                                     "svmlk_sentiment_corenlp_classifier.pkl",
                                     "lk_vectorizer_sentiment_corenlp.pkl", 
                                    "sentiment_corenlp_features.pkl")
        data = TrainingMongoData.tag_data()
        TagClassifiers.svm_bagclassifier(data,
                                     "svmlk_tag_classifier.pkl",
                                     "lk_vectorizer_tag.pkl",
                                     "tag_features.pkl")








