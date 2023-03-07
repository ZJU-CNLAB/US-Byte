# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import socket

DEBUG =0
WARMUP=True
FP16=False

MAX_EPOCHS = 200

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

