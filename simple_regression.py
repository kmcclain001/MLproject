#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:13:31 2018

@author: km3911
"""
import numpy as np
from analysis import *

class SimpleRegresDecoder:
    
    def __init__(self,X,y):
        self.d = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.w = np.zeros(self.d)
        
    def train(self):
        T = self.X.T.dot(self.X)
        self.w = np.linalg.inv(T).dot(self.X.T).dot(self.y)
        return None
        
    def predict(self,X_hat):
        return X_hat.dot(self.w)
        
    def prediction_error(self,X_hat,y_hat):
        pred = self.predict(X_hat)
        return angular_distance(pred*np.pi/180,y_hat*np.pi/180)
    
    def median_error(self,X_hat,y_hat):
        errs = self.prediction_error(X_hat,y_hat)
        return np.median(errs)
    
class RidgeRegresDecoder(SimpleRegresDecoder):
    
    def __init__(self,X,y,lam):
        self.d = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.lam = lam
        self.w = np.zeros(self.d)
    
    def train(self):
        T = self.X.T.dot(self.X)
        self.w = np.linalg.inv(T+self.lam*np.eye(T.shape[0])).dot(self.X.T).dot(self.y)
        return None
    
    def prediction_error(self,X_hat,y_hat):
        pred = self.predict(X_hat)
        return angular_distance(pred*np.pi/180,y_hat*np.pi/180)
    
    def median_error(self,X_hat,y_hat):
        errs = self.prediction_error(X_hat,y_hat)
        return np.median(errs)