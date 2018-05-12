#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:58:12 2018

@author: kathrynmcclain
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from analysis import *

class nnDecoder:
    
    def __init__(self,X,y,fps=30):
        self.n, self.d = X.shape
        self.X = X
        self.y = one_hot(y.astype(int),360)
        self.G = tf.Graph()
        self.fps = fps
        self.training_it = 5000000
        self.training_error = np.zeros(self.training_it)
        with self.G.as_default():
            self.inp = tf.placeholder(shape=[None,self.d],dtype=tf.float32)
            self.y_true = tf.placeholder(shape=[None,360],dtype=tf.float32)
            #W = tf.get_variable('weights',shape=[self.d,360])
            #b = tf.get_variable('bias',shape=[360])
            #self.output = tf.nn.sigmoid(tf.matmul(self.inp,W)+b)
            hidden1 = tf.layers.dense(self.inp,50,activation=tf.nn.sigmoid)
            hidden2 = tf.layers.dense(hidden1,100,activation=tf.nn.sigmoid)
            self.output = tf.layers.dense(hidden2,360,activation=tf.nn.sigmoid)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true,logits=self.output)
            self.mean_ent = tf.reduce_mean(cross_entropy)
            self.train_op = tf.train.GradientDescentOptimizer(.5).minimize(self.mean_ent)
            self.saver = tf.train.Saver()
    
    def train(self,retrain=False):
        self.sess = tf.InteractiveSession(graph=self.G)
        if retrain:
            self.sess.run(tf.global_variables_initializer())
            batch_size = 3*self.fps
            batch_inds = np.arange(0,self.n,batch_size)
        
            for i0 in range(self.training_it):
                if i0%100 == 0:
                    print(i0)
                i = i0%(batch_inds.size-1)
                x_batch = self.X[batch_inds[i]:batch_inds[i+1],:]
                y_batch = self.y[batch_inds[i]:batch_inds[i+1],:]
                #x_batch = self.X
                #y_batch = self.y
                ent,_ = self.sess.run([self.mean_ent,self.train_op], feed_dict={self.inp:x_batch,self.y_true:y_batch})
                self.training_error[i0] = ent
        
            self.saver.save(self.sess,"/home/km3911/Documents/decoder_normalization/trained_nn_decoder0.ckpt")
            plt.figure()
            plt.plot(self.training_error)
        else:
            self.saver.restore(self.sess,"/home/km3911/Documents/decoder_normalization/trained_nn_decoder.ckpt")
            
        
    def predict(self,X_hat):
        outputs = self.sess.run(self.output,feed_dict={self.inp:X_hat})
        y_hat = np.argmax(outputs,axis=1)
        #y_hat = y_hat[:,np.newaxis]
        return y_hat
    
    def prediction_error(self,X_hat,y_hat):
        pred = self.predict(X_hat)
        return angular_distance(pred*np.pi/180,y_hat*np.pi/180)
    
    def median_error(self,X,y_true):
        errs = self.prediction_error(X,y_true)
        return np.median(errs)
    
def one_hot(x,d):
    n = np.max(x.shape)
    R = np.zeros((n,d))
    R[np.arange(n),np.squeeze(x)]=1
    return R