#!/usr/bin/env python
"""
Author: Kaali
Date: 27 April, 2015
Purpose:
    This is the general purpose module to be used by the whole app, for the methods
    who finds their use everywhere


"""


import time
from Text_Processing.colored_print import bcolors
def print_execution(func):
        "This decorator dumps out the arguments passed to a function before calling it"
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        fname = func.func_name
        def wrapper(*args,**kwargs):
                start_time = time.time()
                print "{0} Now {1} of class --<<{2}>>-- have started executing {3}".format(bcolors.OKBLUE, func.func_name, 
                                "Still dont know how to know the name of the class name", bcolors.RESET)
                result = func(*args, **kwargs)
                print "{0} Total time taken by {1} for execution is --<<{2}>>--{3}\n".format(bcolors.OKGREEN, func.func_name,
                                (time.time() - start_time), bcolors.RESET)

                return result
        return wrapper
        
