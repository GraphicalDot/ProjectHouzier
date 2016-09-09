#!/usr/bin/env python
from datetime import date
import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.autoreload 
import tornado.httpserver
import json
from tornado.httpclient import AsyncHTTPClient
import pymongo
import os
import sys
from functools import update_wrapper
from functools import wraps
import time

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_path)



from GlobalConfigs import connection, eateries, reviews, yelp_eateries, yelp_reviews

from FoodDomainApiHandlers.food_word_cloud import FoodWordCloudApiHelper
from FoodDomainApiHandlers.ambience_word_cloud import AmbienceWordCloudApiHelper
from FoodDomainApiHandlers.cost_word_cloud import CostWordCloudApiHelper
from FoodDomainApiHandlers.service_word_cloud import ServiceWordCloudApiHelper

def cors(func, allow_origin=None, allow_headers=None, max_age=None):
        if not allow_origin:
                allow_origin = "*"

        if not allow_headers:
                allow_headers = "content-type, accept"

        if not max_age:
                max_age = 60

        @wraps(func)
        def wrapper(*args, **kwargs):
                response = func(*args, **kwargs)
                cors_headers = {
                                "Access-Control-Allow-Origin": allow_origin,
                                "Access-Control-Allow-Methods": func.__name__.upper(),
                                "Access-Control-Allow-Headers": allow_headers,
                                "Access-Control-Max-Age": max_age,
                                }
                if isinstance(response, tuple):
                        if len(response) == 3:
                                headers = response[-1]
                        else:
                                headers = {}
                        headers.update(cors_headers)
                        return (response[0], response[1], headers)
                else:
                        return response, 200, cors_headers
        return wrapper



class VersionHandler(tornado.web.RequestHandler):
    def get(self):
        print "hello"
        response = { 'version': '3.5.1',
                     'last_build':  date.today().isoformat() }
        self.write(json.dumps(response))
 
class GetGameByIdHandler(tornado.web.RequestHandler):
        def initialize(self, common_string):
                self.common_string = common_string
    
        def get(self, id):
                response = { 'id': int(id),
                     'name': 'Crazy Game',
                     'release_date': date.today().isoformat(), 
                     "common_string": self.common_string,
                     }
                self.write(response)



class GetArgs(tornado.web.RequestHandler):
        def get(self):
                try:
                        eatery_name = self.get_argument("eatery_name")
                except tornado.web.MissingArgumentError:
                        self.set_status(400, "Missing Arhument")
                        
                self.write("success")


class PostArgs(tornado.web.RequestHandler):

        def post(self):
                try:
                        eatery_name = self.get_argument("eatery_name")
                except tornado.web.MissingArgumentError:
                        self.set_status(400, "Missing Arhument")
                        self.write(json.dumps({"messege": "give me the money"}))
                        return 
                self.write("success")
                return


class LimitedEateriesList(tornado.web.RequestHandler):

        """
        This gives only the limited eatery list like the top on the basis of the reviews count
        """
        def initialize(self):
                self.set_header("Access-Control-Allow-Origin",  "*"),
                self.set_header("Access-Control-Allow-Headers", "content-type, accept")
                self.set_header("Access-Control-Max-Age", 60),

        
        
        @tornado.gen.coroutine
        def get(self):
                time.sleep(20)
                result = list(eateries.find(fields= {"eatery_id": True, "_id": False, "eatery_name": True, \
                        "area_or_city": True}).limit(14).sort("eatery_total_reviews", -1))

                for element in result:
                        eatery_id = element.get("eatery_id")
                        element.update({"reviews": reviews.find({"eatery_id": eatery_id}).count()})

                yelp_result = list(yelp_eateries.find(fields= {"eatery_id": True, "_id": False, "eatery_name": True, "area_or_city": True}).limit(5).sort("eatery_total_reviews", -1))

                for element in yelp_result:
                        eatery_id = element.get("eatery_id")
                        element.update({"reviews": yelp_reviews.find({"eatery_id": eatery_id}).count()})
                
                self.write({"success": True,
                        "error": False,
                        "result": result +yelp_result,
                        })



class GetFullPageAsyncHandler(tornado.web.RequestHandler):
        @tornado.gen.coroutine
        def get(self):
                print "handling the request"
                http_client = AsyncHTTPClient()
                http_response = yield http_client.fetch("http://www.drdobbs.com/web-development")
                response = http_response.body.decode().replace("Most Recent Premium Content", "Most Recent Content")
                self.write(response)
                self.set_header("Content-Type", "text/html")

class Login(tornado.web.RequestHandler):
        def get(self):
                items = ["Item 1", "Item 2", "Item 3"]
                self.render("TornadoHtml.html", title="My title", items=items)
                self.write("")

class Application(tornado.web.Application):
        def __init__(self):
                handlers = [
                    (r"/limited_eateries_list", LimitedEateriesList),
                    (r"/getfullpage", GetFullPageAsyncHandler),
                    (r"/getnamebyid/([0-9]+)", GetGameByIdHandler, dict(common_string='Value defined in Application')),
                    (r"/version", VersionHandler),
                    (r"/postargs", PostArgs),
                    (r"/login", Login),
                    (r"/getargs", GetArgs),
                    (r"/images/^(.*)", tornado.web.StaticFileHandler, {"path": "./images"},),
                    (r"/css/(.*)", tornado.web.StaticFileHandler, {"path": "/css"},),
                    (r"/js/(.*)", tornado.web.StaticFileHandler, {"path": "/js"},),]
                settings = dict(cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",)
                tornado.web.Application.__init__(self, handlers, **settings)




def main():
        http_server = tornado.httpserver.HTTPServer(Application())
        tornado.autoreload.start()
        http_server.listen("8000")
        tornado.ioloop.IOLoop.current().start()




if __name__ == "__main__":
        """
        application.listen(8000)
        tornado.autoreload.start()
        tornado.ioloop.IOLoop.instance().start()
        """
        main()




