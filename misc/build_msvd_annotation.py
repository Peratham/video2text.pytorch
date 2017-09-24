"""
Prepare text-related data sets, including:
1. splits the data set
2. transform the captions to tokens
3. prepare ground-truth
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import json
import pickle
from collections import Counter
import nltk
import torch

sys.path.append('/home/jxgu/github/video2text_jxgu/pytorch')
import opts
sys.path.append('../')
from misc.utils import *


if __name__ == '__main__':
    opt = opts.parse_opt()
    build_msvd_annotation(opt)

