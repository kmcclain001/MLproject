#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:57:10 2018

@author: km3911
"""
import numpy as np
from analysis import *
import scipy.io as sio

def make_fake_data(n_cells):
    filt = gaussian_filt(5,sigma=2)
    tuning = np.eye(n_cells)
    for i in range(n_cells):
        tuning[:,i] = np.convolve(filt,tuning[:,i],mode='same')
        
    return np.tile(tuning,(10,1))

class ImportDataCircle:
    """ Datatype that holds different variations of data. You can cut it down
    to get sections you want to look at while still maintaining original data"""
    
    def __init__(self,neur_path,pos_path):
        #Import data from .mat files
        behavior_contents = sio.loadmat(pos_path)
        neural_contents = sio.loadmat(neur_path)
        
        #Put into convenient variables
        pos_raw = behavior_contents['all_position']

        self.trial_times = behavior_contents['trial_times']
        self.trial_types = behavior_contents['trial_types']

        spikes_raw = neural_contents['neural_data']
        
        region_weird = neural_contents['region']
        self.region = np.array([],dtype=object)
        for i in range(region_weird.shape[1]):
            self.region = np.append(self.region,region_weird[0,i][0])
    
        timeline_raw = neural_contents['timeline']
        self.ls_ind = np.where(self.region=='ls')[0]
        self.hpc_ind = np.where(self.region=='hpc')[0]
        
        #filter session - get out times w/o position info
        inds_w_pos = np.array(np.where(~np.isnan(pos_raw[:,0])))
        self.pos_sess = select_point_in_time(pos_raw,inds_w_pos)
        self.spikes_sess = select_point_in_time(spikes_raw,inds_w_pos)
        self.timeline_sess = select_point_in_time(timeline_raw,inds_w_pos)
        
        #filter trials - get data that happens within trials
        inds_trial,self.trial_inds = find_trial_inds(self.trial_times,self.timeline_sess)
        self.n_trials = self.trial_inds.shape[0]
        self.pos = select_point_in_time(self.pos_sess,inds_trial)
        self.spikes = select_point_in_time(self.spikes_sess,inds_trial)
        self.timeline = select_point_in_time(self.timeline_sess,inds_trial)
    
    def convert_position_to_angular(self,center,zero_vec):
        self.center = center
        self.zero_vec = zero_vec
        self.pos_rad = convert_to_angle(self.pos,center,zero_vec)
        self.pos = np.round(self.pos_rad*360/(2*np.pi))%360
        self.pos = self.pos[:,np.newaxis]
        self.compute_trial_direction()
        return None
    
    def remove_silent_cells(self,threshold):
        self.spikes_with_quiet = self.spikes.copy()
        nonsilent_inds = np.where(np.sum(self.spikes,axis=0)>threshold)[0]
        self.spikes = self.spikes[:,nonsilent_inds]
        self.region = self.region[nonsilent_inds]
        
    def smooth_data(self,sigma):
        filt = gaussian_filt(3*sigma,sigma=sigma)
        filt = filt/np.sum(filt)
        for i in range(self.spikes.shape[1]):
            self.spikes[:,i] = np.convolve(filt,self.spikes[:,i],mode='same')
            
    def select_region(self,reg):
        if reg == 'hpc':
            self.spikes = self.spikes[:,self.hpc_ind]
            self.region = self.region[self.hpc_ind]
        elif reg == 'ls':
            self.spikes = self.spikes[:,self.ls_ind]
            self.region = self.region[self.ls_ind]
    
    def compute_trial_direction(self):
        self.trial_directions = trial_directions(self.pos,self.trial_inds)
        
    def select_trial_direction(self,direction):
        correct_dir_trials = self.trial_directions == direction
        wrong_dir_trials = np.where(self.trial_directions != direction)[0]
        correct_dir_inds = concatenate_ranges(self.trial_inds[correct_dir_trials,:])
        
        self.pos = self.pos[correct_dir_inds]
        self.spikes = self.spikes[correct_dir_inds,:]
        self.timeline = self.timeline[correct_dir_inds,:]
        self.trial_directions = self.trial_directions[correct_dir_trials]
        
        for i in range(wrong_dir_trials.size):
            this_trial = wrong_dir_trials[i]
            interval = self.trial_inds[this_trial,1]-self.trial_inds[this_trial,0]
            self.trial_inds[this_trial:,:] = self.trial_inds[this_trial:,:] - interval
        
        self.trial_inds = np.delete(self.trial_inds,wrong_dir_trials,axis=0)
        self.n_trials = np.sum(correct_dir_trials)
        self.trial_times = self.trial_times[correct_dir_trials,:]
        
    def select_cell_inds(self,indices):
        self.spikes = self.spikes[:,indices]
        self.region = self.region[indices]
    
    def add_fps(self,fps):
        self.fps = fps
        