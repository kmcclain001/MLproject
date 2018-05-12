#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:45:30 2018

code to help analyze hippocampal data

@author: mcclain
"""

import numpy as np

def select_point_in_time(data,relevent_times,timeline=None,approx=False):
    """ returns data filted to desired time points
        inputs:
            DATA - data to be filtered, should have each data point as row, 
                    each feature as column
            RELEVENT_TIMES - times that you want to keep, can be list of inds
                                in which case TIMELINE should be None or list
                                of times
            TIMELINE - timeline that corresponds to all data points, if None
                        assume indices have been given
            APPROX - if times in RELEVENT_TIMES are not in TIMELINE, find 
                        closest point
                        
        returns:
            FILTERED_DATA - DATA where only RELEVENT_TIMES have been kept """
            
        
    num_entries_in,num_features = data.shape
    num_entries_out = relevent_times.size
    
    relevent_times = np.reshape(relevent_times,(num_entries_out,1))
        
    filtered_data = np.zeros((num_entries_out,num_features))
    
    if timeline is None:
        indices = relevent_times
        
    else:
        timeline = np.reshape(timeline,(timeline.size,1))
        if approx:
            indices = np.argmin(np.abs(timeline-relevent_times.T),axis=0)
        else:
            relevent_times = relevent_times[np.in1d(relevent_times,timeline)]
            indices = np.where(np.in1d(timeline,relevent_times))
            
    indices = np.squeeze(indices)
    
    if indices.size == 1:
        filtered_data = data[np.newaxis,indices,:]
    else:
        filtered_data = data[indices,:]
        
    return filtered_data

def test_select_point_in_time():
    data = np.array([[1,2,3],[2,3,4],[5,6,7],[7,6,5],[9,1,3]])
    timeline = np.array([10,20,30,40,50])
    rel_ind = np.array([1,3,4])
    rel_exact = np.array([20,30,40])
    rel_approx = np.array([19.5,33,42])
    
    out_ind = data[[1,3,4],:]
    out_exact = data[[1,2,3],:]
    out_approx = data[[1,2,3],:]
    
    test_ind = select_point_in_time(data,rel_ind)
    test_exact = select_point_in_time(data,rel_exact,timeline=timeline)
    test_approx = select_point_in_time(data,rel_approx,timeline=timeline,approx=True)
    
    if np.sum(out_ind==test_ind)==9:
        print('test 1 good')
    if np.sum(out_exact==test_exact)==9:
        print('test 2 good')
    if np.sum(out_approx==test_approx)==9:
        print('test 3 good')
        
    return None

def select_feature(data,feature_numbers):
    """filters data based on selected features
        inputs:
            DATA - data to be filtered, should have each data point as row, 
                    each feature as column
            FEATURE_NUMBERS - features of data that you want to keep
        return:
            FILTERED_DATA - DATA where only FEATURE_NUMBERS have been kept"""
            
    return data[:,feature_numbers]

def angular_distance(x,y,center=None,deg=True):
    """computes distance between two points on a circle
        inputs:
            X - first point, can be angle in radians if CENTER not given, or  
                (x,y) if a CENTER is provided
            Y - second point (same as X)
            CENTER - (optional) center of the circle is working with (x,y) 
                    coords
            DEG - (optional) if true returns distance in degrees, if false
                    returns distance in randians
        returns:
            DIST - angular distance between 2 points in range from 0 to pi/90"""
    if center is None:
        x = x%(2*np.pi)
        y = y%(2*np.pi)
        if x.size>1:
            x1 = x.reshape((x.size))
            y1 = y.reshape((y.size))
            temp = np.zeros((x1.size,2))
            temp[:,0] = np.abs(x1-y1)
            temp[:,1] = 2*np.pi-np.abs(x1-y1)
            dist = np.min(temp,axis=1)
        else:
            dist = np.min(np.abs(x-y),2*np.pi-np.abs(x-y))
    else:
        a = x - center
        b = y - center
        dist = np.arccos((a.dot(b))/(np.linalg.norm(a)*np.linalg.norm(b)))
    
    if deg:
        dist = dist*360/(2*np.pi)
    
    dist = dist.reshape((dist.size,1))
    return dist

def test_angular_distance():
    x_rad1 = np.pi
    y_rad1 = 0
    y_rad2 = -np.pi/2
    
    x_cir = np.array([1,0])
    y_cir = np.array([0,1])
    c = np.array([0,0])
    
    y_cir2 = np.array([0,-1])
    
    rad1 = angular_distance(x_rad1,y_rad1)
    rad2 = angular_distance(x_rad1,y_rad2)
    
    cir1 = angular_distance(x_cir,y_cir,center=c)
    cir2 = angular_distance(x_cir,y_cir2,center=c)
    
    print(rad1)
    print(rad2)
    print(cir1)
    print(cir2)
    
    return None
    
#test_angular_distance()
    
def convert_to_angle(xy,center,zero_deg,ninety=None):
    """
    inputs:
        XY - (x,y position)
        CENTER - center of track in position coordinates
        ZERO_DEG - location of zero degrees (arbitrary)
        NINETY - location of pi/2 (must be pi/2 away from ZERO_DEG), if none,
                    simply rotate ZERO_DEG by pi/2
    returns:
        ANG - angular measure around circle
        """
    vects = xy-center
    zero_vec = zero_deg-center
    
    if ninety is None:
        ninety_vec = np.array([[0,-1],[1,0]]).dot(zero_vec)
    else:
        ninety_vec = ninety-center
        
    cosinvs = np.arccos((vects.dot(zero_vec))/(np.linalg.norm(vects,axis=1)*np.linalg.norm(zero_vec)))
    dot_ninety = vects.dot(ninety_vec)
    lower_half_circle = 2*np.pi*((dot_ninety<0).astype('float'))
    ang = np.abs(cosinvs-lower_half_circle)
    return ang

def test_convert_to_angle():
    xy = np.array([[1,0],[1,1],[-1,-1]])
    center = np.array([0,0])
    zero_deg = np.array([1,0])
    ninety = np.array([0,1])
    
    a = convert_to_angle(xy,center,zero_deg,ninety)
    b = convert_to_angle(xy,center,zero_deg)
    print(a)
    print(b)
    return None

#test_convert_to_angle()
    
def identify_place_cells(raw_tuning,N_spike_thresh=10,peak_thresh=5):
    """
    inputs:
        RAW_TUNING - data set of tuning curves over track with each cell in a 
                        row
        N_SPIKE_THRESH - minimum number of spikes cell must have during session
                        to be considered a place field
        PEAK_THRESH - minimum ratio between peak of place field and average FR
        
    returns:
        PLACE - binary vector that indicates if each cell has a place field
        """
        
    n_cells = raw_tuning.shape[0]
    place = np.ones(n_cells)
    
    low_firing = np.where(np.sum(raw_tuning,axis=1)<N_spike_thresh)
    place[low_firing] = 0
    
    kern = np.array([1,1,1,1])
    kern = kern/np.sum(kern)
    smoothed_field = np.zeros(raw_tuning.shape)
    for i in range(raw_tuning.shape[0]):
        smoothed_field[i,:] = np.convolve(kern,raw_tuning[i,:],mode='same')
        max_i = np.max(smoothed_field[i,:])
        mean_i = np.mean(smoothed_field[i,:])
        if max_i <= peak_thresh*mean_i:
            place[i] = 0
    
    return place

def log_safe(x):
    """
    take log of X with replacement for 0 values
    inputs:
        X - array of data points, should be all nonnegative
        
    output:
        OUT - log(X)
        """
    out = x
    ind = np.abs(x)<1e-10
    out[ind] = 1e-10
    return np.log(out)

def rep_zero(xx,rep):
   """ replace values smaller than rep in x with small value, rep """
   x = xx.copy()
   ind = x==0
   x[ind] =  rep
   return x
   

def find_trial_inds(trial_intervals,timeline):
    """
    inputs:
        TRIAL_INTERVALS - set of start and stop times in trials
        TIMELINE - time point of each indice
        
    output:
        INDS - indices of within trial time points
        """
    inds = np.array([],dtype=int)
    trial_inds = np.zeros((trial_intervals.shape[0],2),dtype=int)
    for i in range(trial_intervals.shape[0]):
        trial_inds[i,0] = inds.size
        interval = trial_intervals[i,:]
        start_ind = np.argmin(np.abs(timeline-interval[0]))
        end_ind = np.argmin(np.abs(timeline-interval[1]))
        inds = np.append(inds,np.arange(start_ind,end_ind))
        trial_inds[i,1] = inds.size
    return inds,trial_inds

def gaussian_filt(half_length,sigma=1):
    filt = np.arange(-half_length,half_length)
    return np.exp((-(filt/sigma)**2)/2)

        
def find_place_field(response, threshold=.2):

    filt = gaussian_filt(3,sigma=2)
    smoothed = np.convolve(filt,response,mode = 'same')
    base = np.median(smoothed)
    peak_val = np.max(smoothed)
    if np.abs(peak_val - base)<1e-5:
        return np.array([0,0])
    
    peak_ind = np.argmax(smoothed)
    thresh = base + threshold*(peak_val-base)
    above_thresh_boo = smoothed>thresh
    crosses = np.where(np.diff(above_thresh_boo)!=0)[0]

    side1 = 0
    side2 = 0
    temp = True
    i = peak_ind
    while temp:
        i -= 1
        if np.in1d(i%response.size,crosses):
            side1 = i
            temp = False

    temp = True
    i = peak_ind
    while temp:
        i +=1
        if np.in1d(i%response.size,crosses):
            side2 = i
            temp = False

    return np.array([side1%response.size,peak_ind,side2%response.size],dtype=int)
    
def extract_activity_in_field(neur_activity,position_data,field,fps):
    
    in_field = np.append([0],np.append(np.in1d(position_data,field),[0]))
    enter_field = np.where(np.diff(in_field)>0)[0]+1
    exit_field = np.where(np.diff(in_field)<0)[0]
    invalid_ind = np.where(exit_field-enter_field<3)[0]
    enter_field=np.delete(enter_field,invalid_ind)
    exit_field=np.delete(exit_field,invalid_ind)

    mean_FR = np.zeros((enter_field.size,neur_activity.shape[1]))

    for i in range(enter_field.size):
        all_spikes = select_point_in_time(neur_activity,np.arange(enter_field[i],exit_field[i]))
        mean_FR[i,:] = fps*np.sum(all_spikes,axis=0)/(exit_field[i]-enter_field[i])
    
    return mean_FR,enter_field,exit_field

def determine_direction(sequence):
    d = np.diff(np.squeeze(sequence))
    n_pos = np.sum(d>0)
    n_neg = np.sum(d<0)
    if n_pos>n_neg:
        return 1
    if n_pos<=n_neg:
        return 0
    
def trial_directions(pos_data,trial_inds):
    directions = np.zeros(trial_inds.shape[0])
    for i in range(trial_inds.shape[0]):
        directions[i] = determine_direction(pos_data[trial_inds[i,0]:trial_inds[i,1]])
        
    return directions

def concatenate_ranges(ranges):
    a = np.array([],dtype=int)
    for i in range(ranges.shape[0]):
        a = np.append(a,np.arange(ranges[i,0],ranges[i,1]))
    return a

def convolve_matrix(M,filt,mode):
    x = np.convolve(filt,M[:,0],mode=mode)
    C = np.zeros((x.size,M.shape[1]))
    for i in range(M.shape[1]):
        C[:,i] = np.convolve(filt,M[:,i],mode=mode)
    return C
    
def remove_nans(list1,list2):
    bad_inds1 = np.where(np.isnan(list1))[0]
    bad_inds2 = np.where(np.isnan(list2))[0]
    bad_inds1 = np.concatenate((bad_inds1, bad_inds2[~np.in1d(bad_inds2,bad_inds1)]))
    outlist1 = list1.copy()
    outlist2 = list2.copy()
    outlist1 = np.delete(outlist1,bad_inds1)
    outlist2 = np.delete(outlist2,bad_inds1)
    return outlist1,outlist2

