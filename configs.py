
reviews_data = dict(
        ip = "localhost",
        port = 27017,
        db = "Reviews",
        eateries= "ZomatoEateries",
        reviews = "ZomatoReviews",
        users = "ZomatoUsers",
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



import pymongo
t_connection = pymongo.MongoClient(training_data["ip"], training_data["port"])
sentiment_collection = t_connection[training_data["db"]][training_data["sentiment"]]



