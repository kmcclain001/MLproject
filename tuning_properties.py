#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:50:22 2018

@author: mcclain
"""

import numpy as np
from analysis import *

class TuningProperties:
    
    def __init__(self, neural_data, pos_data, track_length,fps):
        self.neural_data = neural_data
        self.pos_data = pos_data
        self.n_cells = neural_data.shape[1]
        self.track_length = track_length
        self.occupancy = np.zeros(self.track_length)
        self.mean_response = np.zeros((self.track_length,self.n_cells))
        self.place_fields = np.zeros((self.n_cells,3),dtype=int)
        self.is_place_cell = np.zeros(self.n_cells)
        self.enter_field_inds = np.zeros(self.n_cells,dtype=object)
        self.exit_field_inds = np.zeros(self.n_cells,dtype=object)
        self.fps = fps
    
    def compute_means(self,inds=None):
        internal = False
        if inds is None:
            internal = True
            inds = np.arange(self.pos_data.size)
        neural_section = select_point_in_time(self.neural_data,inds)
        pos_section = select_point_in_time(self.pos_data,inds)
        sum_response_section = np.zeros((self.track_length,self.n_cells))
        occupancy_section = np.zeros(self.track_length)
        for i in range(self.track_length):
            pos_ind = np.where(pos_section==i)[0]
            neur_at_spot = select_point_in_time(neural_section,pos_ind) #spikes
            if neur_at_spot.size>0:
                occ = len(pos_ind)/self.fps #seconds
                sum_response_section[i,:] = np.sum(neur_at_spot,axis=0) #spikes 
                occupancy_section[i] = occ #seconds
        if internal:
            self.mean_response = sum_response_section/occupancy_section[:,np.newaxis]
            self.occupancy = occupancy_section
        return sum_response_section, occupancy_section
    
    def compute_place_fields(self):
        if np.sum(self.mean_response)==0:
            self.compute_means()
        for i in range(self.n_cells):
            f = find_place_field(self.mean_response[:,i])
            self.place_fields[i,:] = f
    
    def define_place_cells(self,min_width,max_width,ave_fr,peak_ratio):
        if np.sum(self.place_fields)==0:
            self.compute_place_fields()
        for i in range(self.n_cells):
            f = self.place_fields[i,:]
            peak = self.mean_response[f[1],i]
            baseline = np.mean(self.mean_response[:,i])
            if f[2]<f[0]:
                width = self.track_length-f[0] + f[2]
            else:
                width = f[2]-f[0]
            if width > min_width and width < max_width and peak > peak_ratio*baseline and baseline>ave_fr:
                self.is_place_cell[i] = 1
        
    def compute_in_field_activity(self,cell_ind):
        if np.sum(self.place_fields)==0:
            self.compute_place_fields()
        f = self.place_fields[cell_ind,:]
        if f[1]<f[0]:
            field = np.arange(f[0],self.track_length)+np.arange(f[1])
        else:
            field = np.arange(f[0],f[1])
        return extract_activity_in_field(self.neural_data,self.pos_data,field)
    
    def select_cell_inds(self,inds):
        self.neural_data = self.neural_data[:,inds]
        self.is_place_cell = self.is_place_cell[inds]
        self.mean_response = self.mean_response[:,inds]
        self.n_cells = inds.size
        self.place_fields = self.place_fields[inds,:]
        
    def normalization_factors(self):
        m = np.mean(self.neural_data,axis=0)
        return self.fps*m