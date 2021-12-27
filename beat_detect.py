
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

import librosa
import librosa.display
import madmom.features.beats as mmb
import pdb

from label import *
from utils import *


    
    
def plotAudioBeats(x, sr, hop_length, onset_bin, beat_sec):
    '''
    Plots mel spectrogram of audio x,
    onset strength and predicted beats in song
    '''
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = np.arange(len(onset_bin)) * (hop_length/sr)
    
    M = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                             y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length,
                             ax=ax[0])
                             
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_bin),
             label='Onset strength')
    ax[1].vlines(beat_sec, 0, 1, alpha=0.5, color='r',
               linestyle='--', label='Beats')
    ax[1].legend()
    plt.show(block=False)
    
    
def createTicks(time_bpm, est_end):
    '''
    Generate ticks based on bpm, offset, and meter
    If more than one bpm/offset section, generate those as well
    Generated ticks are in ms units if specified
    Inputs: time_bpm - list with N elements, each element is a
                       3x1 array with [offset, bpm, meter] info
            est_end - estimated ending of ticks (usually last onset)
    '''
    # get timing ticks
    ticks = np.array([])
    for i, ui in enumerate(time_bpm):
        
        begin_time = ui[0]
        bpm = ui[1]
        meter = ui[2]
        
        if i < len(time_bpm)-1:
            end_time = time_bpm[i+1][0]
        else:
            end_time = est_end

        est_blen = 1/(bpm/60)
        est_tlen = est_blen / meter
        endgap = (end_time - begin_time) % est_tlen
        end_time = end_time - endgap
        arr = np.arange(begin_time, end_time, est_tlen)
        ticks = np.append(ticks,arr)
    
    return ticks
    
    
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
    


def checkTicks(pred_bpm, offset, pred_beats):
    '''
    create ticks in ms based off of predicted bpm and offset
    compare predicted ticks to predicted beats from song
    '''
    # assume meter is 4
    # bpm = 1 / [tick_length] * 1000 * 60
    meter = 4
    beat_ms = (60*1000) / pred_bpm
    tick_ms = beat_ms / meter
    sim_ticks = np.arange(offset, song_sec*1000, tick_ms)
    onset_ms = onset_times * 1000
    
    # visualize ticks and onsets
    #plt.stem(sim_ticks,np.ones(sim_ticks.shape)*0.5,'b', linefmt='--',markerfmt='bo')
    #plt.stem(onset_ms, np.ones(onset_ms.shape), 'g', markerfmt='go')
    #plt.show()
    
    # find nearest tick for every onset
    near_idx, residual = findNearest(sim_ticks, onset_ms)
    # check for overall shifts in onset - tick differences
    # if residuals show a sin wave-like pattern, bpm is wrong
    res_limit_ms = tick_ms / 2
    pdb.set_trace()
    # find large steps in residuals (possible bpm changes in song)
    step_idx = stepDetect(residual[-300:])



def getBPMFromBeats(when_beats):
    '''Find potential bpm from beat positions in seconds'''
    m_res = scipy.stats.linregress(np.arange(len(when_beats)),when_beats)
    first_beat = m_res.intercept 
    beat_step = m_res.slope
    pred_bpm = 60 / beat_step
    # if predicted bpm is very close to integer value, round
    if abs(pred_bpm - round(pred_bpm)) < 0.1:
        pred_bpm = round(pred_bpm)
    # if bpm < 100, double it
    if pred_bpm < 100:
        pred_bpm *= 2
    # if bpm > 400, halve it
    if pred_bpm > 400:
        pred_bpm = pred_bpm / 2
    return pred_bpm



def checkOffsetLabels(sec_len, arr_file, time_bpm):
    '''Use model prediction spectrogram bin outputs to guess bpm range of song'''
    #sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000

    sec_bpm = [x[:] for x in time_bpm]
    for i, section in enumerate(time_bpm):
        sec_bpm[i][0] = section[0]/1000

    new_ticks = createTicks(sec_bpm, sec_len)
    ms_ticks = new_ticks * 1000
    
    # check offset using model predictions
    # requires model predictions
    label_arr, crop_sec = loadModelPred(arr_file, sec_len)

    labels = np.copy(label_arr)
    thresh = 0.1
    labels[labels > thresh] = 1
    labels[labels <= thresh] = 0
    labels = labels.astype(np.uint8)

    # spectrogram bins (default 0.032 seconds)
    N = len(label_arr)
    bin_len = crop_sec / (N-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second
    
    # range of ms shifts to search over
    check_range = np.arange(-50, 51)
    check_sum = np.zeros(check_range.shape)
    for ci, cdiff in enumerate(check_range):
        tbi = np.floor(((ms_ticks+cdiff)/1000) * bin_in_sec).astype(np.int) + 4
        tbi = np.delete(tbi, np.where(tbi >= N))
        check_sum[ci] = labels[tbi].sum()

    best_shift = check_range[np.argmax(check_sum)]
    print('best tick shift: ', best_shift)
    #plt.plot(check_range, check_sum)
    
    # modify offsets to match
    final_sections = [x[:] for x in time_bpm]
    for i, section in enumerate(time_bpm):
        final_sections[i][0] = section[0] + best_shift
        
    print('final time sections: ', final_sections)
    
    return final_sections
    
    
    
def getLibrosaBPM(mp3_file):
    '''Use librosa onset detector to guess BPM range of song'''
    x, sr = librosa.load(mp3_file, sr=None, mono=True)
    song_sec = len(x) / sr
    # lower hop window = higher resolution = more accurate bpm estimation
    # however songs with changing bpms mean too accurate is also bad
    # 256 seems ok for most songs
    hop =  256
    
    onset_frames = librosa.onset.onset_detect(x, sr=sr, hop_length=hop)
    onset_times = onset_frames * (hop / sr)
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
    # osu requires offset value to be in milliseconds
    offset = round(onset_frames[0] * (hop / sr) * 1000)
    
    # if single bpm prediction
    if len(pred_range) == 1:
        pred_bpm = round2Base(pred_range[0], base=5)
    else:
        # assume bpm is whole number, prefer a multiple of 5
        if bpm_conf[1] > bpm_conf[0]:
            pred_5 = roundDown2Base(pred_range[1], base=5)
        else:
            pred_5 = roundUp2Base(pred_range[0], base=5)
            
        if pred_5 >= pred_range[0] and pred_5 <= pred_range[1]:
            pred_bpm = pred_5
        else:
            # resort to using ratio of confidence to nearest integer bpm
            ratio_bpm = (bpm_conf[0] / np.sum(bpm_conf)) * (pred_range[1]-pred_range[0])
            pred_bpm = round(pred_range[0] + ratio_bpm)
        
    print(pred_bpm)
    
    # create ticks in ms based off of predicted bpm and offset
    checkTicks(pred_bpm, offset, onset_times)
    
    return offset, bpm



def getMmbBPM(mp3_file):
    '''Use madmom library to guess BPM of song'''
    x, sr = librosa.load(mp3_file, sr=None, mono=True)
    song_sec = len(x) / sr
    
    # default 100 frames per second
    fps = 160
    bin_len = 1/fps # length of each time bin in seconds
    act = mmb.RNNBeatProcessor(fps=fps)(mp3_file)
    # accurate estimation from here: https://github.com/CPJKU/madmom/issues/416
    when_beats = mmb.BeatTrackingProcessor(fps=fps)(act)
    diff_beats = np.diff(when_beats)
    
    # plot beats and audio
    #plotAudioBeats(x, sr, int(bin_len*sr), act, when_beats)
    
    sections = []
    num_beats = len(when_beats)
    
    # check if entire song is regular
    diff_frames = (diff_beats * fps).astype(np.int)
    median_diff = np.median(diff_frames)
    mask = np.abs(diff_frames - median_diff)
    mask = np.bitwise_or(mask <= 1, np.abs(mask-median_diff) <= 1)
    pdb.set_trace()
    if np.sum(mask) > 0.8 * num_beats:
        pred_bpm = getBPMFromBeats(when_beats)
        sections.append([round(when_beats[0]*1000), pred_bpm, 4])
        return sections
    
    # sliding window over beats in song to create timing sections
    wl = 7
    # pad beats and beat diff arrays with last value
    diff_beats = np.append(diff_beats, np.ones((wl-1,))*diff_beats[-1])
    diff_beats = np.insert(diff_beats, 0, diff_beats[0])
    ld = diff_beats[-1]
    lb = when_beats[-1]
    new_beats = np.arange(lb+ld, lb+(ld*wl), ld)
    when_beats = np.append(when_beats, new_beats)
    wi = 0
    while wi < num_beats:
        wbeats = when_beats[wi:wi+wl]
        wdiffs = diff_beats[wi:wi+wl]
        # mean diff
        mean_diff = np.mean(wdiffs)
        # check if any beat diffs are very different
        irregular = np.abs(wdiffs-mean_diff) > bin_len * 3
        # find bpm for this window
        pred_bpm = getBPMFromBeats(wbeats)
        
        if not irregular.any():
            # this window is regular, skip to end of window
            wi = wi+wl
            # time of section start in ms, predicted bpm, assume meter 4
            sections.append([round(wbeats[0]*1000), pred_bpm, 4])
        else:
            # this window has irregularly spaced beats
            # create new timing section with estimated bpm
            irr_idx = np.where(irregular)[0]
            if irr_idx[0] > wl/4:
                # repredict bpm using regular beats
                pred_bpm = getBPMFromBeats(wbeats[:irr_idx[0]])
                wi = wi + irr_idx[0]
                sections.append([round(wbeats[0]*1000), pred_bpm, 4])
            else:
                wi = wi + wl
                sections.append([round(wbeats[0]*1000), pred_bpm, 4])

    return sections




if __name__=='__main__':
    # get song
    #filename = "Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity]"
    #filename = "Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert]"
    #filename = "YOASOBI - Ano Yume o Nazotte (Sarawatlism) [Daisuki]"
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    #filename = "xi - FREEDOM DiVE (Pikastar) [Universe]"
    #filename = "DM Ashura - Classical Insanity (Louis Cyphre) [Vivacissimo]"
    #filename = "The Quick Brown Fox - The Big Black (Blue Dragon) [WHO'S AFRAID OF THE BIG BLACK]"
    #filename = "YOASOBI - Yoru ni Kakeru (CoLouRed GlaZeE) [Collab Extra]"
    #filename = "AKINO from bless4 & CHiCO with HoneyWorks - MIIRO vs. Ai no Scenario (monstrata) [Tatoe]"
    #filename = "GALNERYUS - RAISE MY SWORD (beem2137) [Feruver's Expert]"
    #filename = "SakiZ - osu!memories (DeRandom Otaku) [Happy Memories]"
    filename = "Wan Ho-Kit, Lee Hon-Kam - Unknown Title (Monstrata) [Let's show them Monstrata's powerful stance.]"
    
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
    #time_bpm = [[0,195,4],[55692,180,4]]
    #time_bpm = cluster fuck mostly [[792,184,4]] sometimes 183-185
    #time_bpm = 180 until 240
    #time_bpm = ~190 and gradually speeds up until ~212 bpm
    
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    wav_file = "songs/osu_mp3/" + filename + ".wav"
    arr_file = "songs/osu_mp3/" + filename + ".npy"
    
    # use hit object predictions from model to get potential bpm ranges
    # the faster the song is and the closer tick diffs are to bin length (0.032 sec),
    # the wider the possible bpm range
    #model_range, model_conf = getModelBPM(mp3_file, arr_file)
    
    # use librosa onset detector to get potential bpm ranges
    #bpm_range, bpm_conf = getLibrosaBPM(mp3_file)
    
    # use rnn beat tracker
    mmb_sections = getMmbBPM(mp3_file)
    
    
    
    
    