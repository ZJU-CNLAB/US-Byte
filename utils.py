import hashlib
import time
import os
import numpy as np
import scipy.stats as stats

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            #os.mkdir(filename)
            os.makedirs(filename)
        except:
            pass

