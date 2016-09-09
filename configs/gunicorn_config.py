#!/usr/bin/env python

import multiprocessing
import os

path = "/Data/applogs"

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
max_requests = 1000
worker_class = 'gevent'
worker_connections = 1000
graceful_timeout = 300
keepalive = 50
debug = True
daemon = True
reload = True
preload = True
accesslog = "{0}/gunicorn_access.log".format(path)
errorlog = "{0}/gunicorn_error.log".format(path)
