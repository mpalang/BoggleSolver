# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:56:01 2026

@author: moritzpalang
"""
import sys


def progress_bar(iteration, total, prefix='', length=50):
    percent = iteration / total
    filled = '#' * int(length * percent)
    empty = '-' * (length - len(filled))
    sys.stdout.write(f'\r{prefix} |{filled}{empty}| {percent:.2%}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')


def timer(function):
    """wrapper that times a function"""
    from time import perf_counter as pc
    from functools import wraps
    @wraps(function)
    def wrapped(*args,**kwargs):
        t0 = pc()
        result = function(*args,**kwargs)
        print(f'{function.__name__} took {(pc()-t0):.2f} s')
        return result
    return wrapped


def over_print(s: str, last= False):
    if last:
        end = None
    else:
        end = ""
    print(f"\r\033[K{s}", end=end, flush=True)


        

