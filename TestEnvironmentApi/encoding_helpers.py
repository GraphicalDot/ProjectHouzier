#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 15 April, 2015
Purpose:
    TO deal with the encoding problems of the reviews
"""


class SolveEncoding:
        def __init__(self):
                pass


        @staticmethod
        def preserve_ascii(obj):
                if not isinstance(obj, unicode):
                        obj = unicode(obj)
                obj = obj.encode("ascii", "xmlcharrefreplace")
                return obj

        @staticmethod
        def to_unicode_or_bust(obj, encoding='utf-8'):
                if isinstance(obj, basestring):
                        if not isinstance(obj, unicode):
                                obj = unicode(obj, encoding)
                return obj

