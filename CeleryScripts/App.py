#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import
from celery import Celery
import os

os.environ.setdefault('CELERY_CONFIG_MODULE', '__Celery_APP.celeryconfig')

app = Celery("__Celery_APP")
# include=["__Celery_APP.ProcessingCeleryTask"])

app.config_from_envvar('CELERY_CONFIG_MODULE')


if __name__ == "__main__":
	app.start()
