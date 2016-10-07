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
from sklearn.feature_extraction.text import CountVectorizer
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
from cPickle import dump, load, HIGHEST_PROTOCOL

from configs import SentimentClassifiersPath, TagClassifiersPath,\
                FoodClassifiersPath, ServiceClassifiersPath,\
                AmbienceClassifiersPath, CostClassifiersPath 


from configs import SentimentVocabularyFileName, SentimentFeatureFileName, SentimentClassifierFileName
from configs import TagVocabularyFileName, TagFeatureFileName, TagClassifierFileName
from configs import FoodVocabularyFileName, FoodFeatureFileName, FoodClassifierFileName
from configs import ServiceVocabularyFileName, ServiceFeatureFileName, ServiceClassifierFileName
from configs import CostVocabularyFileName, CostFeatureFileName, CostClassifierFileName
from configs import AmbienceVocabularyFileName, AmbienceFeatureFileName, AmbienceClassifierFileName



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



def store_with_joblib(file_path, _object, file_name):
        with cd(file_path):
                joblib.dump(__object, file_name)

        return 


def store_with_cPickle(file_path, __object, file_name):
        dump(__object, open('%s/%s'%(file_path,
                                    file_name),
                                'wb'), HIGHEST_PROTOCOL)
        return 



def store_with_hdf5():
        
        return 




class GeneralMethodsClassifiers(object):
        @staticmethod
        def snowball_stemmer(sentences):
                stemmer = SnowballStemmer("english")
                return [stemmer.stem(sent) for sent in sentences]


        @staticmethod 
        def pre_process_text(sentences):
                return [PreProcessText.process(sent) for sent in sentences]

        @staticmethod
        def svm_bagclassifier(data, file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path, bagging=False):
                """
                file_name_classifier: The name under which the joblib must
                                    store the classifier 
                file_name_vectorizer: The name under which the joblib must
                                store the vocabulary of the trained vectorizer
                file_name_features: The combined_features name under which the
                jblib must store th features vector.

                file_path: The filepath at which all these above files must be
                stored
                
                """
                start = time.time()
                tags, sentences=zip(*data)
                sentences = GeneralMethodsClassifiers.snowball_stemmer(sentences)
                sentences = GeneralMethodsClassifiers.pre_process_text(sentences)
                vectorize_class = HouzierVectorizer(sentences,
                                                    file_path, 
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
                                                           tags)
                with cd(file_path):
                        joblib.dump(combined_features, file_name_features, 
                                            compress=("zlib", 9))
                """
                dump(combined_features, open('%s/%s'%(file_path,
                                                      file_name_features),
                                             'wb'), HIGHEST_PROTOCOL)
                """

                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape
                n_estimators=5
                svc_classifier = SVC(kernel='linear', 
                                                            C= 1, 
                                                            gamma="auto", 
                                                            probability=True, 
                                                            decision_function_shape="ovr",
                                                            class_weight="balanced",
                                                            cache_size = 20000
                                     )
                
                
                if bagging:
                        classifier= OneVsRestClassifier(BaggingClassifier(svc_classifier,
                                                        max_samples=1.0,
                                                        max_features= 1.0, 
                                                        n_jobs=-1, 
                                                        verbose=3, 
                                                        n_estimators=n_estimators,
                                                                   bootstrap=False))
                else:
                    classifier = svc_classifier 
                
                
                
                
                classifier.fit(X_features, tags)

                print classifier.classes_
                with cd(file_path):
                        joblib.dump(classifier, file_name_classifier, 
                                            compress=("zlib", 9))
                """
                dump(classifier, open('%s/%s'%(file_path,
                                               file_name_classifier),
                                               'wb'), HIGHEST_PROTOCOL)
                """
                print "Storing Classifier with joblib"
                print time.time() -start
                return 

        @staticmethod
        def svm_bagclassifier_prediction(data, file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path, bagging=False):

        
                target, sentences=zip(*data)
                vectorize_class = HouzierVectorizer(sentences,
                                                    file_path, 
                                                    file_name_vectorizer, False, False)
                #example_counts= example_counts.toarray()
                vocabulary_to_load = vectorize_class.return_vectorizer()
                #vectorize_class = HouzierVectorizer(examples, True, False)
                #x_vectorize = vectorize_class.count_vectorize()
                
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary_to_load) 
                sentences_counts = loaded_vectorizer.transform(sentences)

                with cd(file_path):
                        feature_reduction_class = joblib.load(file_name_features)
                        classifier = joblib.load(file_name_classifier)

                        #feature_reduction_class=load(open(file_name_features, 'rb'))
                        #classifier= load(open(file_name_classifier, 'rb'))
                
                reduced_features = feature_reduction_class.transform(sentences_counts.toarray())


                predictions = classifier.predict(reduced_features)
                print accuracy_score(target, predictions)
                
                
                return 
         



class SentimentClassifiers(object): 


        @staticmethod
        def svm_bagclassifier(sentiment_data, file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              bagging=False):
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
                
                selection = SelectKBest(chi2, k=2)
                combined_features = FeatureUnion([("pca", pca), ("univ_select",
                                                                 selection)])


                X_features = combined_features.fit_transform(X_normalized,
                                                           sentiments)
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(combined_features, file_name_features,
                                    compress=("zlib", 9))
                """
                dump(combined_features,
                     open('%s/%s'%(SentimentClassifiersPath,SentimentFeatureFileName), 'wb'),HIGHEST_PROTOCOL)

                """
                #X_pca = pca.fit_transform(x_transform)


                print "Feature after feature slection with pca and selectkbest\
                    of the data [%s, %s]"%X_features.shape

               
                #http://stackoverflow.com/questions/32934267/feature-union-of-hetereogenous-features

                #clf = SVC(C=1, kernel="linear", gamma=.001, probability=True, class_weight='auto')
                
                n_estimators = 3
                svc_classifier = SVC(kernel='linear', 
                                                            C= 1, 
                                                            gamma="auto", 
                                                            probability=True, 
                                                            decision_function_shape="ovr",
                                                            class_weight="balanced",
                                                            cache_size = 20000
                                     )
                
                
                if bagging:
                        classifier= OneVsRestClassifier(BaggingClassifier(svc_classifier,
                                                        max_samples=1.0,
                                                        max_features= 1.0, 
                                                        n_jobs=-1, 
                                                        verbose=3, 
                                                        n_estimators=n_estimators,
                                                                   bootstrap=False))
                else:
                        classifier = svc_classifier 
                
                
                classifier.fit(X_features, sentiments)




                print classifier.classes_
                with cd("%s/CompiledModels/SentimentClassifiers"%base_dir):
                        joblib.dump(classifier, file_name_classifier,
                                    compress=("zlib", 9))
                """
                dump(file_name_classifier,open('%s/%s'%(SentimentClassifiersPath,
                                                       SentimentClassifierFileName
                                                        ),
                                               'wb'), HIGHEST_PROTOCOL)
                """

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
        def do_gridsearch():
                from sklearn.grid_search import GridSearchCV

               #classifier pipeline
                clf_pipeline = clf_pipeline = OneVsRestClassifier(
                Pipeline([('reduce_dim', RandomizedPCA()),
                          ('clf', classifier())
                          ]
                         ))

                C_range = 10.0 ** np.arange(-2, 9)
                gamma_range = 10.0 ** np.arange(-5, 4)
                n_components_range = (10, 100, 200)
                degree_range = (1, 2, 3, 4)

                param_grid = dict(estimator__clf__gamma=gamma_range,
                      estimator__clf__c=c_range,
                        estimator__clf__degree=degree_range,
                        estimator__reduce_dim__n_components=n_components_range)
                params = dict(C = C_range, gamma = gamma_range)
                clf = GridSearchCV(OneVsRestClassifier(SVC()),params, cv=5,
                                   n_jobs=-1)
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
        def svm_bagclassifier(file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):
                training_data, test_data = TrainingMongoData.tag_data()
                GeneralMethodsClassifiers.svm_bagclassifier(training_data[0:1000],
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                GeneralMethodsClassifiers.svm_bagclassifier_prediction(test_data,
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )

                return 



class CostClassifiers(object):
        
        @staticmethod
        def svm_bagclassifier(file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):
                training_data, test_data =  TrainingMongoData.sub_category_data_cost()
                GeneralMethodsClassifiers.svm_bagclassifier(training_data[0:1000],
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                GeneralMethodsClassifiers.svm_bagclassifier_prediction(test_data,
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )

                return 

class FoodClassifiers(object):
        
        @staticmethod
        def svm_bagclassifier(file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):
                training_data, test_data =  TrainingMongoData.sub_category_data_food()
                GeneralMethodsClassifiers.svm_bagclassifier(training_data[0:1000],
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                GeneralMethodsClassifiers.svm_bagclassifier_prediction(test_data,
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )

                return 

class ServiceClassifiers(object):
        
        @staticmethod
        def svm_bagclassifier(file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):
                training_data, test_data = TrainingMongoData.sub_category_data_service()
                GeneralMethodsClassifiers.svm_bagclassifier(training_data[0:1000],
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                GeneralMethodsClassifiers.svm_bagclassifier_prediction(test_data,
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                return 


class AmbienceClassifiers(object):
        
        @staticmethod
        def svm_bagclassifier(file_name_classifier,
                              file_name_vectorizer, file_name_features,
                              file_path):
                training_data, test_data =  TrainingMongoData.sub_category_data_ambience()
                GeneralMethodsClassifiers.svm_bagclassifier(training_data[0: 1000],
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )
                GeneralMethodsClassifiers.svm_bagclassifier_prediction(test_data,
                                                            file_name_classifier,
                                                            file_name_vectorizer,
                                                            file_name_features,
                                                            file_path
                                                            )

                return 

                



if __name__ == "__main__":
        """
        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_after_corenlp_analysis(),
                                     "svmlk_sentiment_corenlp_classifier.pkl",
                                     "lk_vectorizer_sentiment_corenlp.pkl", 
                                    "sentiment_corenlp_features.pkl")
        data = TrainingMongoData.tag_data()
        TagClassifiers.svm_bagclassifier(data,
                                     "svmlk_tag_classifier.pkl",
                                     "lk_vectorizer_tag.pkl",
                                     "tag_features.pkl")

        """
        SentimentClassifiers.svm_bagclassifier(TrainingMongoData.sentiment_data_three_categories(),
                                               SentimentClassifierFileName,
                                               SentimentVocabularyFileName, 
                                               SentimentFeatureFileName)

        TagClassifiers.svm_bagclassifier(TagClassifierFileName,
                                         TagVocabularyFileName, 
                                         TagFeatureFileName, 
                                        TagClassifiersPath)
        
        CostClassifiers.svm_bagclassifier(
                                        CostClassifierFileName,
                                        CostVocabularyFileName, 
                                        CostFeatureFileName, 
                                        CostClassifiersPath)

        FoodClassifiers.svm_bagclassifier(FoodClassifierFileName,
                                          FoodVocabularyFileName, 
                                          FoodFeatureFileName, 
                                          FoodClassifiersPath
                                          )
        ServiceClassifiers.svm_bagclassifier(ServiceClassifierFileName,
                                            ServiceVocabularyFileName, 
                                             ServiceFeatureFileName, 
                                            ServiceClassifiersPath)

        AmbienceClassifiers.svm_bagclassifier(AmbienceClassifierFileName,
                                              AmbienceVocabularyFileName,
                                              AmbienceFeatureFileName,
                                              AmbienceClassifiersPath)
        





