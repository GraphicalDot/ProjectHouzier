
#!/usr/bin/env python
from os.path import dirname, abspath 
import sys
import pymongo

directory = dirname(abspath(__file__))
root_path = dirname(dirname(dirname(directory)))

sys.path.append(root_path)
from configs import sentiment_collection 


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





if __name__ == "__main__":
        print TrainingMongoData.sentiment_data_three_categories()

