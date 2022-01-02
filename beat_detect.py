
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


    
    
def plotAudioBeats(x, sr, hop_length, onset_bin, beat_sec, plot_mel=False):
    '''
    Plots mel spectrogram of audio x,
    onset strength and predicted beats in song
    '''
    figsize = (10,6)
    if plot_mel:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        ax0 = ax[0]
    else:
        fig, ax = plt.subplots(nrows=1, figsize=figsize)
        ax0 = ax
    
    times = np.arange(len(onset_bin)) * (hop_length/sr)
    
    ax0.plot(times, librosa.util.normalize(onset_bin),
             label='Onset strength')
    ax0.vlines(beat_sec, 0, 1, alpha=0.5, color='r',
               linestyle='--', label='Beats')
    ax0.legend()
    
    if plot_mel:
        M = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=hop_length)
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                                 y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length,
                                 ax=ax[1])

        ax[1].label_outer()
        ax[1].set(title='Mel spectrogram')


def getBPMFromBeats(onset_bins, bin_len, method):
    '''Find potential bpm from beat positions in seconds'''
    if method == 'lr':
        when_beats = onset_bins * bin_len
        m_res = scipy.stats.linregress(np.arange(len(when_beats)),when_beats)
        first_beat = m_res.intercept 
        beat_step = m_res.slope
        pred_bpm = 60 / beat_step
        bpm_range = np.array([pred_bpm-5, pred_bpm+5])
    
    else: # quadratic interpolation
        onset_diffs = np.diff(onset_bins).astype(np.int)
        # sorted unique values and their counts
        uniqvals, counts = np.unique(onset_diffs, return_counts=True)
        est_dist = scipy.interpolate.interp1d(uniqvals, counts, 'quadratic')
        xnew = np.arange(uniqvals[0], uniqvals[-1], 0.01)
        ynew = est_dist(xnew)
        plt.plot(xnew, ynew)
        best_ibin = np.argmax(ynew)
        max_bin = xnew[best_ibin]
        pred_bpm = 60/(max_bin*bin_len)
        # bin peak bases
        irange = int(0.04/bin_len)*100
        pkrange = (max(best_ibin-irange, 0), min(best_ibin+irange, len(ynew)))
        # bpm ranges
        bpm_range = np.array([60/(xnew[pkrange[1]]*bin_len), 
                              60/(xnew[pkrange[0]]*bin_len)])

    # if bpm < 100, double it
    if pred_bpm < 100:
        pred_bpm *= 2
        bpm_range *= 2
    # if bpm > 300, halve it
    if pred_bpm > 300:
        pred_bpm = pred_bpm / 2
        bpm_range = bpm_range / 2
    # round bpm to 2 decimal places
    pred_bpm = np.around(pred_bpm, decimals=2)
    bpm_range = np.around(bpm_range, decimals=2)
    # if predicted bpm is very close to integer value, round
    bpm_shift = pred_bpm - round(pred_bpm)
    if abs(bpm_shift) < 0.05:
        pred_bpm = round(pred_bpm)
        bpm_range = bpm_range - bpm_shift
        
    return pred_bpm, bpm_range
    
    
    
    
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
        
        est_blen = 1/(bpm/60)
        
        if i < len(time_bpm)-1:
            end_time = time_bpm[i+1][0]
        else:
            end_time = est_end

        est_tlen = est_blen / meter
        endgap = (end_time - begin_time) % est_tlen
        end_time = end_time - endgap
        arr = np.arange(begin_time, end_time, est_tlen)
        ticks = np.append(ticks,arr)
    
    return ticks
    
    
def checkTicks(est_x, est_end, onset_times, verbose_f):
    '''
    Check if estimated bpm and offset match with actual song onsets
    Method: Generate the potential beats using estimated bpm and offset
            Match each beat with nearest song onset
            Calculate residual between the beat and nearest onset
            Remove outlier residuals and smooth residual curve
            Return mean abs residual to measure if simulated beat matches onsets
            
    Inputs: est_x - array of size 2, estimated bpm and estimated offset
            est_end - end range of generated ticks from bpm & offset
            onset_times - onset times found through librosa or mmb
                          objective to match with generated ticks
            verbose_f - whether or not to output stats and plot residual curve
    Outputs: res - mean abs residual between nearest onsets and simulated ticks
                   residual signal is smoothed
    '''
    est_bpm = est_x[0]
    est_offset = est_x[1]
    est_blen = 1/(est_bpm/60)
    est_tlen = est_blen/4  # assume meter is 4
    est_beats = np.arange(est_offset, est_end, est_blen)
    num_beats = len(est_beats)
    
    # matches each estimated beat with nearest onset time
    idx, res = findNearest(onset_times, est_beats)
    meanabsres = np.mean(np.abs(res))

    # remove spikes and smooth
    outres = reject_outliers(res, m = 2.)
    outmeanres = np.mean(outres)
    outmeanabsres = np.mean(np.abs(outres))
    outstdres = np.std(outres)
    
    # line fitted to first couple samples (for offset finding)
    first_samps = outres[:int(num_beats/4)]
    first_outs = reject_outliers(first_samps, m = 2.)
    # avg abs offset from 0 for first couple values (for offset finding)
    first_val = np.abs(np.mean(first_outs[:10]))
    
    outresx = np.arange(len(first_outs))
    outresy = first_outs
    lrres = scipy.stats.linregress(outresx,outresy)
    lrintercept = lrres.intercept
    lrslope = lrres.slope
    lrx = np.array([0, len(outres)])
    lry = lrslope * lrx + lrintercept
    lrscore = np.abs(lrslope) + np.abs(lrintercept) + first_val
                     
    #plotres = smooth(outres, win_size=51, method="sg")
    #smoothmeanabsres = np.mean(np.abs(plotres))

    if verbose_f:
        # outputs
        print('est beat length: ', est_blen, ' | est tick length: ', est_tlen)
        print('time range from ', est_offset, ' to ', est_end)
        print('num est beats: ', len(est_beats), ' | num est ticks: ', len(est_ticks))
        print('mean residual: ', np.mean(res), ' | mean abs residual: ', meanabsres)
        print('mean outlier-processed residual: ', outmeanres)
        print('stdev outlier-processed residual: ', outstdres)
        print('mean abs outlier-processed residual: ', outmeanabsres)
        print('lr intercept: ', lrintercept, ' | lr slope: ', lrslope)
        print('lr score: ', lrscore)
        print('first_val: ', first_val)
        
        #print('mean smoothed residual: ', np.mean(plotres))
        #print('mean abs smoothed residual: ', smoothmeanabsres)

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(14,7))
        ax[0].plot(res)
        ax[1].plot(outres)
        ax[1].plot(lrx,lry,'-r')
    
    return outstdres, lrscore, outmeanabsres


def searchCombs(combs, xtype, metric, verbose_f, args):
    '''
    Search over bpm and offset combinations 
    to find best matching bpm and offset
    Inputs: combs - Nx2 array with N combinations
                    of bpm (col 0) and offset (col 1)
            metric - col of result metrics to optimize
            args - other arguments for checkTicks func
    '''
    
    results = []
    for cx in combs:
        cres = checkTicks(cx, *args)
        cres = np.array(cres)
        results.append(cres[np.newaxis,:])

    resarr = np.vstack(results)
    mc = resarr[:,metric]
    # normalize and average columns if more than one metric
    if len(metric) > 1:
        mc = (mc - np.min(mc,axis=0)) / (np.max(mc,axis=0) - np.min(mc,axis=0))
        mc = np.mean(mc, axis=1)
    bestx = combs[np.argmin(mc),:]
    
    if verbose_f:
        print('searching over ', combs.shape[0], ' combinations')
        
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
        ax[0].plot(combs[:,xtype], resarr[:,0])
        ax[1].plot(combs[:,xtype], resarr[:,1])
        ax[2].plot(combs[:,xtype], resarr[:,2])

        print('best [bpm, offset]: ', bestx)
        print('best metric: ', np.min(mc))
    
    return bestx



def searchLoop(bpm_bounds, offset_bounds, x0, args):
    '''Search over bpm and offset until convergence'''
    verbose_f = 0
    
    est_bpm = x0[0]
    est_offset = x0[1]
    est_blen = 1/(est_bpm/60)
    est_tlen = est_blen/4
    
    # rough search (0.1 bpm increments)
    potential_bpm = np.arange(bpm_bounds[0], bpm_bounds[1], 0.1)
    potential_offset = [est_offset]
    print('searching bpm range: ', bpm_bounds, ' with offset ', est_offset, ' and increment ', 0.1)
    combs = np.array(np.meshgrid(potential_bpm, potential_offset)).T.reshape(-1,2)
    # search using std of residual to narrow down bpm range
    bestcomb = searchCombs(combs, 0, [0], verbose_f, args)
    print('best comb: ', bestcomb)
    
    # search for offset (0.001 sec increments)
    potential_bpm = [bestcomb[0]]
    potential_offset = np.arange(offset_bounds[0], offset_bounds[1], 0.001)
    print('searching offset range: ', offset_bounds, ' with bpm ', bestcomb[0], ' and increment ', 0.001)
    combs = np.array(np.meshgrid(potential_bpm, potential_offset)).T.reshape(-1,2)
    # search using closeness of linear regression line to 0 slope and 0 intercept
    bestcomb2 = searchCombs(combs, 1, [0, 1], verbose_f, args)
    print('best comb: ', bestcomb2)
    
    # fine tuning
    # measure mean abs residual between predicted beats and librosa onsets
    used_metric = [2]
    
    # finer search (0.01 bpm increments)
    new_bounds = (bestcomb[0]-0.09, bestcomb[0]+0.09)
    potential_bpm = np.arange(new_bounds[0], new_bounds[1], 0.01)
    potential_offset = [bestcomb2[1]]
    print('searching bpm range: ', new_bounds, ' with offset ', bestcomb2[1], ' and increment ', 0.01)
    combs = np.array(np.meshgrid(potential_bpm, potential_offset)).T.reshape(-1,2)
    bestcomb3 = searchCombs(combs, 0, used_metric, verbose_f, args)
    final_bpm = bestcomb3[0]
    print('best comb: ', bestcomb3)
    
    # search for offset (0.001 sec increments)
    new_bounds = (max(0, bestcomb2[1]-est_tlen), bestcomb2[1]+est_tlen)
    potential_bpm = [final_bpm]
    potential_offset = np.arange(new_bounds[0], new_bounds[1], 0.001)
    print('searching offset range: ', new_bounds, ' with bpm ', final_bpm, ' and increment ', 0.001)
    combs = np.array(np.meshgrid(potential_bpm, potential_offset)).T.reshape(-1,2)
    bestcomb4 = searchCombs(combs, 1, used_metric, verbose_f, args)
    final_offset = bestcomb4[1]
    print('best comb: ', bestcomb4)
    
    plot_test = 1
    if plot_test:
        new_args = [args[0], args[1], 1]
        checkTicks(bestcomb, *new_args)
        checkTicks(bestcomb2, *new_args)
        checkTicks(bestcomb3, *new_args)
        checkTicks(bestcomb4, *new_args)
    
    return final_bpm, final_offset



def checkOffsetLabels(mp3_file, arr_file, time_bpm):
    '''Use model prediction spectrogram bin outputs to guess bpm range of song'''
    sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000
    
    # check offset using model predictions
    # requires model predictions
    label_arr, crop_sec = loadModelPred(arr_file, sec_len)
    
    # spectrogram bins (default 0.032 seconds)
    N = len(label_arr)
    bin_len = crop_sec / (N-1)  # length of each time bin in seconds
    bin_in_sec = 1 / bin_len  # number of bins in every second

    #labels = np.copy(label_arr)
    #thresh = 0.1
    #labels[labels > thresh] = 1
    #labels[labels <= thresh] = 0
    #labels = labels.astype(np.uint8)
    
    sec_bpm = [x[:] for x in time_bpm]
    for i, section in enumerate(time_bpm):
        sec_bpm[i][0] = section[0]/1000
    
    new_ticks = createTicks(sec_bpm, sec_len)
    ms_ticks = new_ticks * 1000
    
    best_shifts = getBestShifts(time_bpm, ms_ticks, label_arr, bin_in_sec, N, crop_sec)
    
    final_sections = [x[:] for x in time_bpm]
    for i, section in enumerate(final_sections):
        section[0] = section[0] + best_shifts[i]
        
    print('final time sections: ', final_sections)
    
    return final_sections
    
    
    
def getLibrosaBPM(mp3_file):
    '''
    Use librosa onsets library to guess BPM of song
    Returns: list of tuples (offset, bpm, meter) 
             for each section of song with different bpm
    '''
    x, sr = librosa.load(mp3_file, sr=None, mono=True)
    song_sec = len(x) / sr
    
    # librosa onset
    hop = 256   
    onset_frames = librosa.onset.onset_detect(x, sr=sr, hop_length=hop)
    onset_times = onset_frames * (hop / sr)
    o_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop)
    
    # plot onset strength
    #plt.figure(figsize=(14,7))
    #times = np.arange(len(o_env)) * (hop/sr)
    #plt.plot(times, librosa.util.normalize(o_env), label='Onset strength')
    #plt.vlines(onset_times, 0, 1, alpha=0.5, color='r',
    #           linestyle='--', label='onsets')
    
    # quadratic interpolation of beat frame diff histogram
    quad_bpm, bpm_range = getBPMFromBeats(onset_frames, (hop / sr), 'quad')
    print("quad estimated bpm: ", quad_bpm)
    print("quad estimated bpm range: ", bpm_range)
    
    # simulate ticks using estimated bpm and offset
    est_bpm = quad_bpm
    est_offset = onset_times[0]
    est_end = min(onset_times[-1]+1, song_sec)

    est_blen = 1/(est_bpm/60)
    est_tlen = est_blen/4

    # specify search bounds for bpm and offset
    x0 = np.array([est_bpm, est_offset])
    bpm_bounds = [est_bpm-10, est_bpm+10]
    offset_bounds = (max(0, est_offset-est_tlen*2), est_offset+est_tlen*2)

    print("initial bpm, initial offset: ", x0)
    print("ticks end (seconds): ", est_end)
    print("bpm search range: ", bpm_bounds)
    print("offset search range: ", offset_bounds)
    
    # arguments for checkTicks function
    args = (est_end, onset_times, 0)
    # search over bounds to find optimal bpm and offset
    final_bpm, final_offset = searchLoop(bpm_bounds, offset_bounds, x0, args)
    
    # double check offset using different hop length
    #hop = int(sr/100)
    #onset_frames2 = librosa.onset.onset_detect(x, sr=sr, hop_length=hop)
    #onset_times2 = onset_frames2 * (hop / sr)
    #new_bounds = (max(0, final_offset-0.05), final_offset+0.05)
    #potential_bpm = [final_bpm]
    #potential_offset = np.arange(new_bounds[0], new_bounds[1], 0.001)
    #print('searching offset range: ', new_bounds, ' with bpm ', final_bpm, ' and increment ', 0.001)
    #combs = np.array(np.meshgrid(potential_bpm, potential_offset)).T.reshape(-1,2)
    #bestcomb = searchCombs(combs, 1, used_metric, 0, (est_end, onset_times2, 0))
    #final_offset = bestcomb[1]
    
    # finalize outputs
    final_offset = np.around(final_offset, decimals=3)
    final_offset = int(final_offset*1000)
    final_bpm = np.around(final_bpm, decimals=2)
    
    # if predicted bpm is very close to integer value, round
    bpm_shift = final_bpm - round(final_bpm)
    if abs(bpm_shift) < 0.06:
        final_bpm = round(final_bpm)
    
    # final results
    print('[offset bpm meter]: ', final_offset, final_bpm, 4)
    
    time_bpm = []
    # always assume meter 4
    # offset in ms
    time_bpm.append([final_offset, final_bpm, 4])
    
    return time_bpm



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
    #filename = "umu. - humanly (Half) [Len's Robotically Another]"
    #filename = "Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity]"
    #filename = "Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert]"
    #filename = "YOASOBI - Ano Yume o Nazotte (Sarawatlism) [Daisuki]"
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    filename = "xi - FREEDOM DiVE (Pikastar) [Universe]"
    #filename = "DM Ashura - Classical Insanity (Louis Cyphre) [Vivacissimo]"
    #filename = "Die For You ft. Grabbitz  - VCT 2021"
    #filename = "Train - 50 Ways to Say Goodbye"
    #filename = "Nightcore - Dernière Danse"

    #filename = "The Quick Brown Fox - The Big Black (Blue Dragon) [WHO'S AFRAID OF THE BIG BLACK]"
    #filename = "YOASOBI - Yoru ni Kakeru (CoLouRed GlaZeE) [Collab Extra]"
    #filename = "AKINO from bless4 & CHiCO with HoneyWorks - MIIRO vs. Ai no Scenario (monstrata) [Tatoe]"
    #filename = "GALNERYUS - RAISE MY SWORD (beem2137) [Feruver's Expert]"
    #filename = "SakiZ - osu!memories (DeRandom Otaku) [Happy Memories]"
    #filename = "Wan Ho-Kit, Lee Hon-Kam - Unknown Title (Monstrata) [Let's show them Monstrata's powerful stance.]"

    # [Time(ms), bpm, meter(beats per measure)]
    # bpm = 1 / [tick_length] * 1000 * 60

    # check if predicted offset and BPM are close to actual song values
    #time_bpm = [[983,250,4]]
    #time_bpm = [[15688, 175, 4]]
    #time_bpm = [[540,91,4],[2245,89,4]]
    #time_bpm = [[1342,180,4]]
    #time_bpm = [[-30,200,4]]
    #time_bpm = [[2133,222.22,4]]
    #time_bpm = [[38,175,4],[64152,175,3],[75466,175,4]]
    #time_bpm = [[5620,190,4]]
    #time_bpm = [[430,140,4]]
    #time_bpm = [[590,132.25,4]]

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
    time_bpm = getLibrosaBPM(mp3_file)
    print(time_bpm)
    
    final_sections = checkOffsetLabels(mp3_file, arr_file, time_bpm)
    
    ### Experiment 1: Create map from audio file and predicted labels

    sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000
    m_empty = Map.fromTiming(final_sections,mp3_file,mp3_len=mp3_len)
    #m_empty = Map.fromTiming([[2133,222.22,4]],mp3_file,mp3_len=mp3_len)

    label_arr, crop_sec = loadModelPred(arr_file, sec_len)

    obj_words = m_empty.encodeHitLabels2Map(crop_sec, label_arr)
    m_empty.decodeWords(obj_words)
    m_empty.saveMap2Osu()
    
    # use rnn beat tracker
    #mmb_sections = getMmbBPM(mp3_file)
    
    
    
    
    