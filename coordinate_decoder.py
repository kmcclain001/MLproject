#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:49:19 2018

beginning the decoder!

@author: km3911
"""
import numpy as np
from data_processing import *
from simple_regression import *
from matplotlib import pyplot as plt
from tuning_properties import *
from bayesian_decoder import *
from nn_decoder import *

## IMPORT DATA
data = ImportDataCircle('DT2_neural_data_py.mat','DT2_behavioral_data_py.mat')
data.convert_position_to_angular(np.array([314.661,249.047]),np.array([485.732,251.896]))
data.add_fps(30)
data.select_region('hpc')
data.smooth_data(3)
data.select_trial_direction(0)
data.remove_silent_cells(1)

neural_data = data.spikes
pos_data = data.pos

## DIVIDE INTO TEST/TRAIN
n = neural_data.shape[0]
data_inds = np.arange(n)
np.random.shuffle(data_inds)
train_inds = data_inds[:int(n*3/5)]
test_inds = data_inds[int(n*3/5):]
neural_train = neural_data[train_inds,:]
pos_train = pos_data[train_inds]
neural_test = neural_data[test_inds,:]
pos_test = pos_data[test_inds]

## INSTANTIATE DECODER
decode = nnDecoder(neural_train,pos_train)

## TRAIN DECODER
decode.train()

## TRAINING ERROR
me_train = decode.median_error(neural_train,pos_train)

## MAKE PREDICTION ON TEST DATA
pos_pred = decode.predict(neural_test)

## TEST ERROR
me_test = decode.median_error(neural_test,pos_test)

## COMPARISON ERROR
sham_inds = test_inds.copy()
np.random.shuffle(sham_inds)
neural_sham = neural_data[sham_inds,:]
np.random.shuffle(sham_inds)
pos_sham = pos_data[sham_inds]
me_sham = decode.median_error(neural_sham,pos_sham)

## VISUALIZE
print('train error/sham '+str(me_train/me_sham))
print('test error/sham '+str(me_test/me_sham))