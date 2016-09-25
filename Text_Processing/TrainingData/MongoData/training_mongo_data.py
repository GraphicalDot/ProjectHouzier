
#!/usr/bin/env python
from os.path import dirname, abspath 
import sys
import pymongo

directory = dirname(abspath(__file__))
root_path = dirname(dirname(dirname(directory)))

sys.path.append(root_path)
from configs import sentiment_collection, corenlp_collection 


class TrainingMongoData(object):
        """
        This class is used to get training data stored in the mongodb 
        be it meant for sentiment analysis, food sub tag analysis and
        so on 
        """
        

        @staticmethod
        def sentiment_data_three_categories():
                """
                returns a list of the form (sentiment, sentence)
                only have three categories positive, neutral, negative 
                """
            
                sentiments = [(post.get("sentiment"), post.get("sentence")) for
                              post in sentiment_collection.find()]
                return [(sentiment.split("-")[-1], sentence) for sentiment, sentence
                  in set(sentiments)]



        @staticmethod
        def sentiment_data_five_categories():
                sentiments = [(post.get("sentiment"), post.get("sentence")) for
                          post in sentiment_collection.find()]
                return list(set(sentiments))




        @staticmethod
        def sentiment_data_after_corenlp_analysis():
                """
                The data generated by this function will be as follows, 
                The training_data.sentiment collection have a parsed key which
                have stanford corenlp analysis of the sentences
                For each sentence this methiod will replace NNS NNP DT and CC 
                part of speech tag with a dummy variable so as to decrease the 
                vocubulary size for the classifiers
                """
                def each_sentences(parsed_result):
                        part_of_speech_list = list()
                        for element in parsed_result["sentences"][0]["words"]:
                                part_of_speech_list.append((element[0], element[1]["PartOfSpeech"]))

                        sentence = list()
                        for (word, pos) in part_of_speech_list:
                                if pos in ["DT", "NNS", "NNP", "CC"]:
                                        word = "DUMMY"
                                sentence.append(word)
                        return " ".join(sentence)
                    
                sentiments = [(post.get("sentiment"),
                               each_sentences(post.get("parsed"))) for post in
                              corenlp_collection.find()]

                result = list(set(sentiments))
                print result
                return result


if __name__ == "__main__":
        cls = TrainingMongoData.sentiment_data_after_corenlp_analysis()
        print cls




