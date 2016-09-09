#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 3rd February, 2015
Purpose: This module has been written to take care of the part of speech tagging of the sentences
"""
import os

from pos_tagging import PosTaggers

PosTaggerDirPath = os.path.dirname(os.path.abspath(__file__))
HunPosModelPath = '{0}/hunpos-1.0-linux/en_wsj.model'.format(PosTaggerDirPath)
HunPosTagPath = '{0}/hunpos-1.0-linux/hunpos-tag'.format(PosTaggerDirPath)
