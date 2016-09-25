

import pymongo
import os
from os.path import dirname, abspath

base_dir = dirname(abspath(__file__)) 
print base_dir

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)





reviews_data = dict(
        ip = "localhost",
        port = 27017,
        db = "Reviews",
        eateries= "ZomatoEateries",
        reviews = "ZomatoReviews",
        users = "ZomatoUsers",
)



corenlp_data = dict(
        ip = "localhost",
        port = 3456,
        db = "corenlp",
        sentiment= "sentiment",
        path_jar_files = "/Users/kaali/Programs/Python/ProjectHouzier/stanford-corenlp-python"
)


training_data = dict(
        ip = "localhost",
        port = 27017,
        db  = "training_data",
        sentiment = "training_sentiment_collection",
        food = "training_food_collection",
        service = "training_service_collection",
        ambience = "training_ambience_collection",
        cost = "training_cost_collection",
        tag = "training_tag_collection",
)

results_data = dict(
        ip = "localhost",
        port = 27017,
        db = "results",
        reviews = "reviews",
        eateries = "eateries"
    )


celery = dict(
        celery_redis_broker_ip = "localhost",
        celery_redis_broker_port = 6379,
        celery_redis_broker_db_number = 0,
)

debug = dict(
        all = True,
        results = False,
        execution_time = True,
        print_docs = False,
    )



t_connection = pymongo.MongoClient(training_data["ip"], training_data["port"])
sentiment_collection = t_connection[training_data["db"]][training_data["sentiment"]]
corenlp_collection = t_connection[corenlp_data["db"]][corenlp_data["sentiment"]]


import sys
sys.path.append(corenlp_data["path_jar_files"])
import jsonrpc
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=(corenlp_data["ip"],
                                                          corenlp_data["port"]
                                                          )))






