import os
import re
import sys
import pdb
import numpy as np
from math import log, e
from scipy.signal import find_peaks

import word
from utils import *

# hit object subword integer representations from word.py
h_int = word.obj_str2int[word.HITCIRCLE]
e_int = word.obj_str2int[word.EMPTY]
sb_int = word.obj_str2int[word.SLIDER_BEGIN]
sc_int = word.obj_str2int[word.SLIDER_CENTER]
se_int = word.obj_str2int[word.SLIDER_END]
b_int = word.obj_str2int[word.BREAK]

SW_int = word.dir_str2int[word.SW]
SE_int = word.dir_str2int[word.SE]
NW_int = word.dir_str2int[word.NW]
NE_int = word.dir_str2int[word.NE]
W_int = word.dir_str2int[word.W]
E_int = word.dir_str2int[word.E]
N_int = word.dir_str2int[word.N]

crawl_int = word.vel_str2int[word.CRAWL]
slow_int = word.vel_str2int[word.SLOW]
med_int = word.vel_str2int[word.MED]
    
    
    
def loadModelPred(arr_file, sec_len):
    '''
    arr_file: path to numpy array predictions saved by model
    sec_len: length of original song in seconds
    '''
	# process label arr (from model prediction)
    label_arr = np.load(arr_file)
    num_bins = len(label_arr)
    
    # cropped audio length (from spectrogram calculations) in seconds
    crop_sec = int(np.floor(sec_len*16000/512)*512)/16000
    
    return label_arr, crop_sec
  
  
def setSlider(tick_obj, word_arr, ss, se):
    tick_obj[ss] = sb_int
    tick_obj[se] = se_int
    slider_dir = np.random.choice(np.array([NW_int, N_int, NE_int]))
    if se - ss > 1:
        tick_obj[ss+1:se] = sc_int
        word_arr[ss:se+1,1] = slider_dir
        word_arr[ss+1:se,2] = crawl_int
        word_arr[se,2] = med_int
    else:
        word_arr[se,1] = slider_dir
        word_arr[se,2] = med_int
 

def getTickBins(tarr, bin_in_sec, N):
        # shift four bins into the future due to spectrogram window being 4 hop lengths (2048/512)
        tbi = np.around((tarr/1000) * bin_in_sec, decimals=0).astype(np.int) + 4
        tbi = np.delete(tbi, np.where(tbi >= N))
        return tbi


def getBestShifts(time_bpm, ms_ticks, label_arr, bin_in_sec, N, crop_sec):
    
    # modify offsets to match
    best_shifts = []
    for i, section in enumerate(time_bpm):
        if i == len(time_bpm)-1:
            rg = (section[0], crop_sec*1000)
        else:
            rg = (section[0], time_bpm[i+1][0])
        
        section_ticks = ms_ticks[np.bitwise_and(ms_ticks >= rg[0], ms_ticks < rg[1])]
        
        # range of ms shifts to search over
        check_range = np.arange(-200, 201)
        check_sum = np.zeros(check_range.shape)
        for ci, cdiff in enumerate(check_range):
            new_ticks = section_ticks+cdiff
            new_ticks = new_ticks[new_ticks >= 0]
            tbi = getTickBins(new_ticks, bin_in_sec, N)
            check_sum[ci] = label_arr[tbi].sum()

        # smooth sums to find peak of curves
        sumsmooth = smooth(check_sum, win_size=31, method="sg")
        pkidx, pkdict = scipy.signal.find_peaks(sumsmooth, prominence=np.ptp(sumsmooth)/10)
        pkshifts = check_range[pkidx]
        best_shift = check_range[pkidx[np.argmin(np.abs(pkshifts))]]

        choice = 0
        if choice == 1:
            import matplotlib.pyplot as plt
            plt.plot(check_range, check_sum)
            plt.plot(check_range, sumsmooth)
            plt.vlines(best_shift, np.min(sumsmooth), np.max(sumsmooth),
                       alpha=0.5, color='r', linestyle='--', label='best shift')
        print('best tick shift: ', best_shift)
        
        best_shifts.append(best_shift)
    
    return best_shifts
    
    
        
def label2Array(label_arr, tick_arr, time_bpm, wav_len):
    '''
    labels: array Nx1 with values [0,1] indicating probability of hit object
    tick_arr: ticks in ms
    time_bpm: list of lists, with each element list containing 
                    [offset, bpm, meter] for each uninherited timing section
    wav_len: length of cropped audio from spectrogram in seconds
    '''
    ticks = np.copy(tick_arr)
    # set all objects greater than threshold to 3 (hitcircle) for now
    labels = np.copy(label_arr)
    thresh = 0.1
    labels[labels > thresh] = h_int
    labels[labels <= thresh] = e_int
    labels = labels.astype(np.uint8)
    objs = labels == h_int
    
    N = len(labels)
    bin_len = wav_len / (N-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    # convert ticks (ms) to time bin indices
    #tick_diff = np.diff(ticks)
    # only keep tick idx with difference > bin length
    #kept_ticks = np.where(tick_diff > round(bin_len*1000))[0]
    #kept_ticks = ticks[kept_ticks]
    
    # search for best model predictions for given timing ticks
    choice = 1
    if choice == 1:
        best_shifts = getBestShifts(time_bpm, ticks, label_arr, bin_in_sec, N, wav_len)
        for i, section in enumerate(time_bpm):
            if i == len(time_bpm)-1:
                rg = (section[0], wav_len*1000)
            else:
                rg = (section[0], time_bpm[i+1][0])
            section_idx = np.bitwise_and(ticks >= rg[0], ticks < rg[1])
            ticks[section_idx] = ticks[section_idx] + best_shifts[i]
            
        tbi = getTickBins(ticks, bin_in_sec, N)
        
    else:
        tbi = getTickBins(ticks, bin_in_sec, N)
        
    # if too many hit objects, increase threshold
    while objs[tbi].sum() > len(objs)/4:
        if thresh >= 0.95:
            break
        thresh += 0.05
        labels[label_arr > thresh] = h_int
        labels[label_arr <= thresh] = e_int
        objs = labels == h_int
    # if too few hit objects, decrease threshold
    while objs[tbi].sum() < len(objs)/10:
        if thresh <= 0.05:
            break
        thresh -= 0.05
        labels[label_arr > thresh] = h_int
        labels[label_arr <= thresh] = e_int
        objs = labels == h_int
        
    # get final hit objects for each tick
    tick_obj = labels[tbi]
    
    # initialize word array
    word_arr = np.zeros((len(tick_obj), 3)).astype(np.uint8)
    word_arr[:,0] = tick_obj.flatten()
    word_arr[:,1] = E_int
    word_arr[:,2] = slow_int
    
    # now look for potential slider starts and ends
    # two consecutive hitcircles: slider start and end
    hits = np.copy(tick_obj).flatten()
    # object index
    h_idx = np.where(hits == h_int)[0]
    # object mask
    h_mask = np.zeros(h_idx.shape, dtype=bool)
    # difference in ticks between each hit object
    diff = np.diff(h_idx)
    
    
    
    
    ### JUMPS: find distribution of tick differences (exclude high tick diff)
    diff_dist, _ = np.histogram(diff[diff<10], bins=np.arange(11))
    # most common tick difference is assumed to be the
    # base time difference between hitcircle jumps
    # exclude 1 tick difference
    jump_diff = np.argmax(diff_dist[2:])+2
    # mask of all ticks with constant base tick diff
    base_mask = diff == jump_diff
    # jump sections have constant base tick diff for longer than 4 objects
    jump_starts = []
    jump_mask = np.copy(h_mask)
    jump_areas = pattwhere_sequ([True,True,True], base_mask)
    if jump_areas.any():
        jump_idx_list = consecutive(jump_areas)
        # store starting tick idx and length of every jump section (hit circles > 4)
        for jump_idx in jump_idx_list:
            sec_len = len(jump_idx)
            # section length + 2 from the extension of [True,True,True] window
            tup = (jump_idx[0], sec_len+2)
            jump_starts.append(tup)
            jump_mask[tup[0]:tup[0]+tup[1]+1] = True
    
    # jumps should have changing direction (either SW or SE)
    # jumps should have medium velocity
    for jtup in jump_starts:
        s_hidx = jtup[0]
        s_hlen = jtup[1]
        # first hitcircle in jump section won't change velocity
        jump_idx = h_idx[s_hidx+1:s_hidx+s_hlen+1]
        word_arr[jump_idx,1] = np.random.choice(np.array([W_int, E_int, SW_int, SE_int]))
        word_arr[jump_idx,2] = med_int
        # break up long jump sections with sliders
        limit = np.random.randint(6,11)
        if s_hlen > limit:
            num_breaks = s_hlen // limit
            ss_idx = np.arange(s_hidx+limit, s_hidx+s_hlen, limit)
            for ss in ss_idx:
                setSlider(tick_obj, word_arr, h_idx[ss], h_idx[ss+1])
    
    
    
    ### STREAMS: store starting tick idx and length of every stream (> 3 consec hitcircles)
    stream_starts = []
    stream_mask = np.copy(h_mask)
    # find all occurrences of two consecutive hitcircles
    twos = pattwhere_sequ([h_int, h_int], hits)
    if twos.any():
        # 2 consecutive twos = 3 hitcircles, 3 consec twos = 4 hitcircles, etc
        twos_idx_list = consecutive(twos)
        for twos_idx in twos_idx_list:
            tup_num = len(twos_idx)
            # >= 3 consec hitcircles (stream)
            if tup_num > 1:
                # store tuple (stream starting tick index, stream length in ticks)
                tup = (twos_idx[0], tup_num+1)
                stream_starts.append(tup)
                stream_obj_mask = np.bitwise_and(h_idx >= tup[0], h_idx < tup[0]+tup[1])
                stream_mask[stream_obj_mask] = True
    
    # streams should have a constant direction (either NW or NE)
    # streams should have 'c' velocity (unless spaced streams are wanted)
    for stup in stream_starts:
        sidx = stup[0]
        slen = stup[1]
        # first hitcircle in stream won't change direction or velocity
        word_arr[sidx+1:sidx+slen,1] = np.random.choice(np.array([NW_int, N_int, NE_int]))
        word_arr[sidx+1:sidx+slen,2] = np.random.choice(np.array([crawl_int, slow_int]))
    
    
    
    ### SLIDERS: hit objects not belonging to jump or stream sections
    slider_mask = ~(jump_mask | stream_mask)
    slider_idx = h_idx[slider_mask]
    slider_diff = np.diff(slider_idx)
    # gaps less than threshold (10 ticks) can be made sliders
    obj_avail = np.where(slider_diff < 11)[0]
    # use every other obj as slider start
    slider_starts = slider_idx[obj_avail[::3]]
    slider_ends = slider_idx[obj_avail[::3]+1]
    # sliders should have a changing direction (NW, N, NE)
    # slider centers should have 'c' velocity
    # slider ends should have 'm' slider velocity
    for ss, se in zip(slider_starts, slider_ends):
        setSlider(tick_obj, word_arr, ss, se)
        
    
    ### remove lone objects
    #circle_idx = tick_obj == 3
    #for ci in circle_idx:
    #    print(ci)
    
    # final update to word_arr
    word_arr[:,0] = tick_obj.flatten()
    
    # visualize word_arr
    #import matplotlib.pyplot as plt
    #plt.plot(word_arr[:,0])
    #plt.plot(word_arr[:,1])
    #plt.plot(word_arr[:,2])
    #plt.show()
    #pdb.set_trace()
    
    return word_arr




def array2Label(tick_arr, arr, wav_len, num_bins):
    '''
    tick_arr: ticks in ms
    word_arr: [num ticks] x 3 array with hitobject, direction, and velocity information
    wav_len: length of cropped audio from spectrogram in seconds
    num_bins: number of bins in spectrogram
    '''
    N = num_bins
    bin_len = wav_len / (num_bins-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    labels = np.zeros((N,3))
    
    # convert ticks (ms) to time bin indices
    tbi = np.floor((tick_arr/1000) * bin_in_sec).astype(np.int)
    # shift four bins into the future due to spectrogram window being 4 hop lengths (2048/512)
    tbi = tbi + 4
    # hit object classes
    labels[tbi,:] = arr
    
    return labels