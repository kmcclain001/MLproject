#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:13:26 2018
Bayesian Decoder
@author: km3911
"""
import numpy as np
from tuning_properties import *

class BayesianDecoder():
    
    def __init__(self,X,y,fps):
        self.d = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.fps = fps
        
    def train(self):
        tuning = TuningProperties(self.X,self.y,360,self.fps)
        tuning.compute_means()
        self.prior = tuning.occupancy
        self.f = tuning.mean_response
        
    def predict(self,X_hat):
        y_hat = np.zeros((X_hat.shape[0],1))
        for i in range(X_hat.shape[0]):
            n = X_hat[i,:]
            t1 = np.prod(self.f**n,axis=1)
            t2 = np.exp(-(1/self.fps)*np.sum(self.f,axis=1))
            post = self.prior*t1*t2
            post = post/np.sum(post)
            y_hat[i] = np.argmax(post)
        return y_hat
    
    def prediction_error(self,X_hat,y_hat):
        pred = self.predict(X_hat)
        return (pred-y_hat)**2
    
    def MSE(self,X_hat,y_hat):
        errs = self.prediction_error(X_hat,y_hat)
        return np.mean(errs)