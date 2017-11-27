# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:55:01 2017

@author: nemes
"""
import scipy
from scipy import linalg, matrix
def null(A, eps=1e-8):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)