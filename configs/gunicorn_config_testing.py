#!/usr/bin/env python

import multiprocessing
import os
import eventlet.debug
deventlet.debug.hub_exceptions(True)


path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/applogs")
print path

bind = "localhost:8000"
workers = multiprocessing.cpu_count() * 2 + 1
access_logformat = "[dev.api] %(h)s %(l)s %(u)s %(t)s .%(r)s. %(s)s %(b)s .%(f)s. .%(a)s. conn=\'%({Connection}i)s\'"

max_requests = 1000
#worker_class = 'eventlet'
#worker_class = 'gevent'
worker_connections = 1000
graceful_timeout = 300
keepalive = 50
debug = True
daemon = False
reload = True
preload = True
threads = 12
log_level = "debug"
