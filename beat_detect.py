
import numpy as np
import matplotlib.pyplot as plt
import madmom.features.beats as mmb
import pdb

import librosa
from util.bpm import getBPM
from label import *


def round2Base(x, base=10):
    return base * round(x/base)
    
    
    
def getBPMRange(bin_len, onset_bin_idx):
    '''Returns potential BPM range of song using binned onset indices'''
    
    onset_diffs = np.diff(onset_bin_idx).astype(np.int)
    # sorted unique values and their counts
    uniqvals, counts = np.unique(onset_diffs, return_counts=True)
    count_perc = counts / sum(counts)
    # get top k unique values
    k = 5
    pos_idx = counts.argsort()[-k:]
    pos_idx.sort()
    pos_bins = uniqvals[pos_idx]
    pos_perc = count_perc[pos_idx]
    maxbin = uniqvals[np.argmax(counts)]
    
    print(uniqvals)
    print(counts)
    
    # get only highest count consecutive to most common bin diff
    consec_bins = consecutive(pos_bins)
    for arr in consec_bins:
        if maxbin in arr:
            maxidx = list(arr).index(maxbin)
            if len(arr) > 2:
                arr = arr[max(0,maxidx-1):min(len(arr), maxidx+2)]
            if len(arr) == 3:
                # 3 consecutive bin diffs, compare counts
                next_count = pos_perc[pos_bins == arr[2]]
                prev_count = pos_perc[pos_bins == arr[0]]
                if next_count >= prev_count:
                    arr = arr[1:]
                else:
                    arr = arr[:2]
                    
            # get proportions of chosen bin diffs as well
            if len(arr) > 1:
                maxidx = list(arr).index(maxbin)
                bmax = np.argmax(counts)
                if maxidx == 0: 
                    pos_perc = count_perc[bmax:bmax+2]
                else: 
                    pos_perc = count_perc[bmax-1:bmax+1]
            else:
                pos_perc = max(count_perc)
            # found maxbin, break from loop
            pos_bins = arr
            break
            
    assert(len(pos_bins) < 3)
    print(pos_bins)
    
    # assume meter is multiple of 4
    # only one possible bpm size
    pos_beats = [16, 8, 4, 2, 1, 1/2, 1/4, 1/8, 1/16]
    sec_ticks = pos_bins * bin_len
    pos_sec_beats = np.outer(pos_beats, sec_ticks)
    pos_bpms = (1 / pos_sec_beats) * 60
    # potential bpm ranges from 50 to 470 bpms
    # if tick length is 0.032 seconds and meter = 4, bpm = 1/(0.032*4)*60 = 468.75
    
    # assuming bpm within 100 to 300 range
    pred_row = np.argmax(np.sum(np.bitwise_and(pos_bpms > 100, pos_bpms < 300), axis=1))
    pred_range = np.flip(pos_bpms[pred_row,:])
    
    # if two columns
    if len(pred_range) > 1:
        print('predicted bpm range: ', pred_range[0], ' to ', pred_range[1])
        pos_bpms, pos_perc = np.flip(pos_bpms,axis=1), np.flip(pos_perc)
        print('predicted bpm confidence: ', pos_perc[0], ' and ', pos_perc[1])
    else:
        print('predicted bpm range: ', pred_range[0])
        print('predicted bpm confidence: ', pos_perc)
    
    return pred_range, pos_bpms, pos_perc
    
    
    

def getModelBPM(mp3_file, arr_file):
    '''Use model prediction spectrogram bin outputs to guess bpm range of song'''
    sec_len = librosa.get_duration(filename=mp3_file)
    labels, wav_len = loadModelPred(arr_file, sec_len)
    N = len(labels)
    bin_len = wav_len / (N-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    # bins where model is sure of prediction
    onsets = np.where(labels > 0.5)[0]
    # predicted offset by model based on first hitobject bin
    offset = onsets[0]*bin_len
    return getBPMRange(bin_len, onsets)
    
    
    
def getLibrosaBPM(mp3_file):
    '''Use librosa onset detector to guess BPM range of song'''
    x, sr = librosa.load(mp3_file, sr=None, mono=True)
    # lower hop window = higher resolution = more accurate bpm estimation
    # however songs with changing bpms mean too accurate is also bad
    # 256 seems ok for most songs
    hop =  256
    
    onset_frames = librosa.onset.onset_detect(x, sr=sr, hop_length=hop)
    #onset_times = onset_frames * (hop / sr)
    o_env = librosa.onset.onset_strength(x, sr=sr)
    # plot onset strength
    # plt.plot(o_env)
    # plt.plot(onset_frames, o_env[onset_frames])
    # plt.show()
    
    bin_len = hop / sr  # length of each time bin in seconds
    
    # get potential ranges of bpm and confidence of each limit
    pred_range, bpm_range, bpm_conf = getBPMRange(bin_len, onset_frames)
    
    # simulate ticks starting from first onset
    # measure goodness of fit with onsets
    # create new bpm section if needed
    # get accurate first onset estimate using small hop length
    hop = 64
    onset_frames2 = librosa.onset.onset_detect(x, sr=sr, hop_length=hop)
    # osu requires offset value to be in milliseconds
    offset = round(onset_frames2[0] * bin_len * 1000)
    
    # assume bpm is whole number, prefer a multiple of 10
    
    #pred_bpm = 
    
    return offset, bpm



def getMmbBPM(mp3_file):
    '''Use madmom library to guess BPM of song'''
    act = mmb.RNNBeatProcessor()(song_file)
    proc = mmb.DBNBeatTrackingProcessor(max_bpm=400, fps=100)
    pred = proc(act)
    b_diff = np.diff(pred)
    
    # check results
    print(pred.shape)
    print('offset: ', pred[0]*1000)
    print(b_diff*1000)
    print('median beat len:',np.median(b_diff)*1000)
    print('mean beat len:',np.mean(b_diff)*1000)
    print('std beat len:',np.std(b_diff)*1000)
    
    #bpm = 1 / [tick_length] * 1000 * 60
    median_beat_sec = np.median(b_diff)*1000
    pred_bpm = 1 / median_beat_sec * 60
    print('predicted median bpm: ', pred_bpm)
    pdb.set_trace()
    return pred_bpm




if __name__=='__main__':
    # get song
    #filename = "Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity]"
    filename = "Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert]"
    #filename = "YOASOBI - Ano Yume o Nazotte (Sarawatlism) [Daisuki]"
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    #filename = "xi - FREEDOM DiVE (Pikastar) [Universe]"
    #filename = "DM Ashura - Classical Insanity (Louis Cyphre) [Vivacissimo]"
    #filename = "The Quick Brown Fox - The Big Black (Blue Dragon) [WHO'S AFRAID OF THE BIG BLACK]"
    #filename = "YOASOBI - Yoru ni Kakeru (CoLouRed GlaZeE) [Collab Extra]"
    
    # [Time(ms), bpm, meter(beats per measure)]
    # bpm = 1 / [tick_length] * 1000 * 60
    
    # check if predicted offset and BPM are close to actual song values
    #time_bpm = [[15688, 175, 4]]
    #time_bpm = [[540,91,4],[2245,89,4]]
    #time_bpm = [[1342,180,4]]
    #time_bpm = [[-30,200,4]]
    #time_bpm = [[2133,222.22,4]]
    #time_bpm = [[38,175,4],[64152,175,3],[75466,175,4]]
    #time_bpm = [[6966,360.3,4]]
    #time_bpm = [[1040,130,4]]
    
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    wav_file = "songs/osu_mp3/" + filename + ".wav"
    arr_file = "songs/osu_mp3/" + filename + ".npy"
    
    # use hit object predictions from model to get potential bpm ranges
    # the faster the song is and the closer tick diffs are to bin length (0.032 sec),
    # the wider the possible bpm range
    #model_range, model_conf = getModelBPM(mp3_file, arr_file)
    
    
    # use librosa onset detector to get potential bpm ranges
    # outperforms model onsets due to
    # 1. higher sampling rate of original audio
    # 2. ability to decrease hop length to get higher resolution
    bpm_range, bpm_conf = getLibrosaBPM(mp3_file)
    
    pdb.set_trace()
    
    # use DBN beat tracker
    mmb_bpm = getMmbBPM(mp3_file)
    
    
    
    
    