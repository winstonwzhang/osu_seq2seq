import os
import re
import sys
import pdb
import numpy as np
from math import log, e

import word


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


def pattwhere_sequ(pattern, a):
    # find index of all occurrences of subarray pattern in array a
    pattern, a = map(np.asanyarray, (pattern, a))
    k = len(pattern)
    if k>len(a):
        return np.empty([0], int)
    hits = np.flatnonzero(a == pattern[-1])
    for p in pattern[-2::-1]:
        hits -= 1
        hits = hits[a[hits] == p]
    return hits


def consecutive(data, stepsize=1):
    # in: np.array([0,47,48,51,52,53])
    # out: [array([0]), array([47,48]), array([51,52,53])]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def entropy(arr, base=None):
  """ Computes entropy of array distribution. """
  n_labels = len(arr)
  if n_labels <= 1:
      return 0
  value,counts = np.unique(arr, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
      return 0
  ent = 0.
  # Compute entropy
  base = e if base is None else base
  for i in probs:
      ent -= i * log(i, base)

  return ent
    
    
def label2WordArray(label_arr, tick_arr, wav_len):
    '''
    labels: array Nx1 with values [0,1] indicating probability of hit object
    tick_arr: ticks in ms
    wav_len: length of cropped audio from spectrogram in seconds
    '''
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
    
    # set all objects greater than threshold to 3 (hitcircle) for now
    labels = np.copy(label_arr)
    labels[labels > 0.1] = h_int
    labels[labels <= 0.1] = e_int
    labels = labels.astype(np.uint8)
    
    N = len(labels)
    bin_len = wav_len / (N-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    # convert ticks (ms) to time bin indices
    tbi = np.floor((tick_arr/1000) * bin_in_sec).astype(np.int)
    # shift four bins into the future due to spectrogram window being 4 hop lengths (2048/512)
    tbi = tbi + 4
    tbi = np.delete(tbi, np.where(tbi >= N))
    # take average of bin values around each tick?
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
    # lower means streams, higher means 
    base_diff = np.argmax(diff_dist)
    # mask of all ticks with constant base tick diff
    base_mask = diff == base_diff
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
        word_arr[jump_idx,1] = np.random.choice(np.array([SW_int, SE_int]))
        word_arr[jump_idx,2] = med_int
    
    
    
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
        word_arr[sidx+1:sidx+slen,1] = np.random.choice(np.array([NW_int, NE_int]))
        word_arr[sidx+1:sidx+slen,2] = crawl_int
    
    
    
    ### SLIDERS: hit objects not belonging to jump or stream sections
    slider_mask = ~(jump_mask | stream_mask)
    slider_idx = h_idx[slider_mask]
    slider_diff = np.diff(slider_idx)
    # gaps less than threshold (10 ticks) can be made sliders
    obj_avail = np.where(slider_diff < 11)[0]
    # use every other obj as slider start
    slider_starts = slider_idx[obj_avail[::2]]
    slider_ends = slider_idx[obj_avail[::2]+1]
    # sliders should have a changing direction (NW, N, NE)
    # slider centers should have 'c' velocity
    # slider ends should have 'm' slider velocity
    for ss, se in zip(slider_starts, slider_ends):
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




def wordArray2Label(tick_arr, word_arr, wav_len, num_bins):
    '''
    tick_arr: ticks in ms
    word_arr: [num ticks] x 3 array with hitobject, direction, and velocity information
    wav_len: length of cropped audio from spectrogram in seconds
    num_bins: number of bins in spectrogram
    '''
    N = num_bins
    bin_len = wav_len / (num_bins-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    labels = np.zeros((N,), dtype=np.uint8)
    
    # convert ticks (ms) to time bin indices
    tbi = np.floor((tick_arr/1000) * bin_in_sec).astype(np.int)
    # shift four bins into the future due to spectrogram window being 4 hop lengths (2048/512)
    tbi = tbi + 4
    # hit object classes
    labels[tbi] = word_arr[:,0]
    
    return labels