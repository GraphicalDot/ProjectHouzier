#!/usr/bin/env python
#-*- coding: utf-8 -*-


from __future__ import absolute_import
import base64
import copy
import re
import csv
import codecs
import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.autoreload
from tornado.httpclient import AsyncHTTPClient
from tornado.log import enable_pretty_logging
import hashlib
import subprocess
import shutil
import json
import os
import StringIO
import difflib
from textblob.np_extractors import ConllExtractor 
from bson.json_util import dumps

from compiler.ast import flatten
from topia.termextract import extract
import decimal
import time
from datetime import timedelta
import pymongo
from collections import Counter
from functools import wraps
import itertools
import random
from multiprocessing import Pool
import base64
import requests
import inspect
import functools
import tornado.httpserver
from itertools import ifilter
from tornado.web import asynchronous
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from bson.son import SON
from termcolor import cprint 
from pyfiglet import figlet_format
from Crypto.PublicKey import RSA
import jwt
from jwt import _JWTError
import ConfigParser



server_address = "localhost"
file_path = os.path.dirname(os.path.abspath(__file__))
parent_dirname = os.path.dirname(os.path.dirname(file_path))






##genrates public and private key everytime you restarts server
os.chdir(parent_dirname)
subprocess.call(["openssl", "genrsa", "-out", "private.pem", "1024"])
subprocess.call(["openssl", "rsa", "-in", "private.pem", "-out", "public.pem", "-outform", "PEM", "-pubout"])
os.chdir(file_path)

private = open("%s/private.pem"%parent_dirname).read()
public = open("%s/public.pem"%parent_dirname).read()
private_key = RSA.importKey(private)
public_key = RSA.importKey(public)

def print_execution(func):
        "This decorator dumps out the arguments passed to a function before calling it"
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        fname = func.func_name
        def wrapper(*args,**kwargs):
                start_time = time.time()
                print "Now {0} have started executing".format(func.func_name)
                result = func(*args, **kwargs)
                print "Total time taken by {0} for execution is --<<{1}>>--\n".format(func.func_name,
                                (time.time() - start_time))
                return result
        return wrapper




def cors(f):
        @functools.wraps(f) # to preserve name, docstring, etc.
        def wrapper(self, *args, **kwargs): # **kwargs for compability with functions that use them
                self.set_header("Access-Control-Allow-Origin",  "*")
                self.set_header("Access-Control-Allow-Headers", "content-type, accept")
                self.set_header("Access-Control-Max-Age", 60)
                return f(self, *args, **kwargs)
        return wrapper
                                                        





def httpauth(arguments):
        def real_decorator(func):
                def wrapped(self, *args, **kwargs):
                        token =  self.get_argument("token")
                        print arguments
                        try:
                                header, claims = jwt.verify_jwt(token, public_key, ['RS256'])
                                self.claims = claims
                                self.messege = None
                                for __arg in arguments:
                                        try:
                                                claims[__arg]
                                        except Exception as e:
                                                self.messege = "Missing argument %s"%__arg
                                                self.set_status(400)
                        except _JWTError:
                                self.messege = "Token expired"
                                self.set_status(403)
                        except Exception as e:
                                self.messege = "Some error occurred"
                                print e
                                self.set_status(500)
                            
                        
                        if self.messege:
                                self.write({
                                        "error": True,
                                        "success": False, 
                                        "messege": self.messege, 
                                })
                                
                        return func(self, *args, **kwargs) 
                return wrapped                   
        return real_decorator


class GetKey(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        def post(self):
                    if self.get_argument("secret") != '967d2b1f6111a198431532149879983a1ad3501224fb0dbf947499b1':
                            self.write({
                                "error": False,
                                "success": True, 
                                "messege": "Key, Nahi milegi", 
                                })
                            self.finish()
                            return 

                    self.write({
                                "error": True,
                                "success": False, 
                                "result": private, 
                        })
                    self.finish()
                    return 


class Test(tornado.web.RequestHandler):
	@cors
	@tornado.gen.coroutine
	@asynchronous
        @print_execution
        @httpauth(["latitude", "longitude"])
        def post(self):
                if not self.messege:
                        self.__on_response()
                self.finish()
                return 

        def __on_response(self):
                time.sleep(10)
                self.write({"success": True,
                            "error": False, 
                            "result": "success",
                            })
                return 

class GetApis(tornado.web.RequestHandler):
        @cors
	@print_execution
	@tornado.gen.coroutine
        def post(self):
                """
                Args:
                        skip:
                        limit:
                        fb_id:

                result:
                    array of dicts
                            each object:
                                
                """
                if self.get_argument("key") != '967d2b1f6111a198431532149879983a1ad3501224fb0dbf947499b1':
                            self.write({
                                "error": False,
                                "success": True, 
                                "messege": "api, Nahi milegi", 
                                })
                            self.finish()
                            return 
                result = {
                        "suggestions": "{0}/{1}".format(server_address, "suggestions"), 
                        "textsearch": "{0}/{1}".format(server_address, "textsearch"), 
                        "getkey": "{0}/{1}".format(server_address, "getkey"), 
                        "userprofile": "{0}/{1}".format(server_address, "userprofile"), 
                        "gettrending": "{0}/{1}".format(server_address, "gettrending"), 
                        "nearesteateries": "{0}/{1}".format(server_address, "nearesteateries"), 
                        "usersdetails": "{0}/{1}".format(server_address, "usersdetails"), 
                        "usersfeedback": "{0}/{1}".format(server_address, "usersfeedback"), 
                        "writereview": "{0}/{1}".format(server_address, "writereview"), 
                        "fetchreview": "{0}/{1}".format(server_address, "fetchreview"), 
                        "geteatery": "{0}/{1}".format(server_address, "geteatery"),
                        }

                self.write({
                            "error": True, 
                            "success": False, 
                            "result": result,
                            })
                self.finish()
                return 


app = tornado.web.Application([
                    (r"/test", Test),
                    (r"/getkey", GetKey),
                    (r"/apis", GetApis),])

def main():
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.bind("8888")
        enable_pretty_logging()
        http_server.start(0) 
        loop = tornado.ioloop.IOLoop.instance()
        loop.start()


"""

r = requests.post("http://52.76.176.188:8000/getkey", data={"secret": '967d2b1f6111a198431532149879983a1ad3501224fb0dbf947499b1'})
key = r.json()["result"]
private_key = RSA.importKey(key)
 token = jwt.generate_jwt({"latitude": "28.5538388889", "longitude": "77.1945111111"}, private_key, 'RS256', datetime.timedelta(minutes=5))
 r = requests.post("http://localhost:8000/test", data={"token": token})
"""
if __name__ == '__main__':
    cprint(figlet_format('Server Reloaded', font='big'), attrs=['bold'])
    main()
