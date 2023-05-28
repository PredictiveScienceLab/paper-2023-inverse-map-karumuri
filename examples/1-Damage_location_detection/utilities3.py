import torch
import torch.nn as nn
import numpy as np
import operator
from functools import reduce
import pickle

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n 
        self.mean = torch.mean(x, 0) # n
        self.std = torch.std(x, 0) # n
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


# saving needed objects

def save_object(obj, filename):
    fileObj = open(filename, 'wb')
    pickle.dump(obj,fileObj, pickle.HIGHEST_PROTOCOL)
    fileObj.close()      
    
def load_object(filename):
    fileObj = open(filename, 'rb')
    obj = pickle.load(fileObj)
    fileObj.close()  
    return obj