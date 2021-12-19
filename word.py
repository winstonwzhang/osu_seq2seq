import pdb
import math
import random
import numpy as np
import pandas as pd  # working with strings

from math_tools import *


# set random seed for deterministic results
random.seed(2020)

# default center of osu play area
DEFAULT_X = 256
DEFAULT_Y = -192  # treating osu y axis as negative for my sanity
DEFAULT_DIREC = np.array([0, 1])  # up direction (cardinal N, north)

# osu play area limits
X_LIM = [0, 512]
Y_LIM = [-384, 0]

# always use normal hitsounds for now
DEFAULT_HITSOUND = 0
DEFAULT_EDGESOUND = "0"
DEFAULT_EDGESET = "0:0"
DEFAULT_HITSAMPLE = "0:0:0:0"


r22 = np.sqrt(2)/2

class Word:
    '''
    "Words" give object type, direction, and velocity information for hitobjects
    Special words for slider beginning and end and slider repeats
    Also encodes ticks with no events (no objects) and breaks
    
    Vocab:
    - 1st subword [h,slb,slc,sle,spin,b,e]
        hitcircle, hitcircle double, slider begin, slider center,
        slider end, spinner, break, empty
    - 2nd subword [E, NE, N, NW, W, SW, S, SE]
        direction of tick vector (with previous tick direction being N)
    - 3rd subword [c, s, m, f]
        distance between prev obj and current obj divided by # ticks passed since previous obj
        crawl, slow, med, fast: magnitude of distance per tick needed
        >= 0, >= 50, >= 150, >= 300 osu!pixels per tick
    
    Examples:
    "h_N_c"    - tick is a hit circle, direction did not change from before, very slow
                 speed needed to hit (either stacked on previous, or empty time in between)
    "slb_S_f"  - tick is at start of slider, direction reversed from previous direction. There
                 is a significant jump or distance between previous hit object and this slider.
    "slc_NE_c" - tick is in middle of slider, direction changed to NE. 3rd subword is ignored
                 since slider speed is constant (decoding uses 3rd subword of slider end [sle])
    "spin"       - tick is during a spinner
    
    # # # # # # # # # # # # # # # # # # # # # # # #
    Example encoding of 237768 Our Stolen Theory - United (LAOS Remix) (Asphyxia) [Infinity]
    Link to beatmap: https://osu.ppy.sh/beatmapsets/237768
    Excerpt from 3 min 07 sec 288 ms to 3 min 08 sec 745 ms
    
    tick 0-5: 'slb_N_c', 'slc_N_c', 'slc_N_c', 'slc_N_c', 'sle_N_s', 'e'
    tick 6-11: 'slb_NW_c', 'slc_NW_c', 'sle_N_c', 'e', 'slb_SE_s', 'slc_SW_c'
    tick 12-17: 'sle_N_c', 'e', 'h_SW_s', 'spin', 'spin', 'spin'
    
    Ticks 0 to 4: Slider start is close to previous hitobject and moving in same direction as before.
                  Slider is a straight line (constant 'N' along slider ticks). Slider end tells us
                  the slider speed is between 30 and 60 osu!pixels per tick ('s').
    Tick 5: 'e' tells us there is no hitobject at this specific tick.
    Ticks 6 to 8: Slider start has shifted ~45 degrees counterclockwise from previous slider end ('NW').
                  Middle of slider also curves ~45 degrees 'NW' from slider start. Slider only lasts
                  for 3 ticks in time. Slider speed is between 0 and 30 osu!pixels per tick ('c').
    Tick 14: hit circle is 'SW' of previous slider end. Hit circle is placed 140-300 osu!pixels away from
             previous slider end (distance/2 is between 70 and 150 since subword 3 is 's').
    Ticks 15 to 17: A spinner is happening at these ticks.
    # # # # # # # # # # # # # # # # # # # # # # # #
    
    Total corpus size from combinations of subwords:
    3 [h,slb,sle] x 8 [E,NE,N,NW,W,SW,S,SE] x 4 [c,s,m,f] = 96
    1 [slc] x 8 [E,NE,N,NW,W,SW,S,SE] x 1 [c]             = 8
    3 [spin,b,e]                                          = 3
    96 + 8 + 3 = 107 unique words
    '''
    # object subwords
    HITCIRCLE = 'h'
    HITCIRCLE_DOUBLE = 'hd'
    SLIDER_BEGIN = 'slb'
    SLIDER_CENTER = 'slc'
    SLIDER_END = 'sle'
    SPINNER = 'spin'
    BREAK = 'b'
    EMPTY = 'e'
    # object string to int dictionary
    obj_str2int = dict([(y,x) for x,y in enumerate(['e','b','spin','h','slb','slc','sle'])])
    
    # cardinal direction: radian thresholds
    t_NNE = np.pi/8
    t_NEE = np.pi*3/8
    t_ESE = np.pi*5/8
    t_SES = np.pi*7/8
    # cardinal direction: subwords
    E = 'E'
    NE = 'NE'
    N = 'N'
    NW = 'NW'
    W = 'W'
    SW = 'SW'
    S = 'S'
    SE = 'SE'
    # cardinal direction: rotation [cos(theta) sin(theta)] vals
    rot = {'E': [0.0, -1.0],
           'NE': [r22, -r22],
           'N': [1.0, 0.0],
           'NW': [r22, r22],
           'W': [0.0, 1.0],
           'SW': [-r22, r22],
           'S': [-1.0, 0.0],
           'SE': [-r22, -r22]
           }
    # direction string to int dictionary
    dir_str2int = dict([(y,x) for x,y in enumerate(['E','NE','N','NW','W','SW','S','SE'])])
    
    # velocity: osu!pixel per tick thresholds (non-slider)
    t_CRAWL = 0
    t_SLOW = 40
    t_MED = 90
    t_FAST = 180
    # velocity: osu!pixel per tick thresholds (slider speed)
    t_sCRAWL = 0  # sliderMultiplier 0.0 hecto-osupix assuming meter of 4
    t_sSLOW = 30  # sliderMultiplier 1.2 hecto-osupix assuming meter of 4
    t_sMED = 60   # sliderMultiplier 2.4 hecto-osupix assuming meter of 4
    t_sFAST = 90  # sliderMultiplier 3.6 hecto-osupix assuming meter of 4
    t_s = {'c': t_sCRAWL, 's': t_sSLOW, 'm': t_sMED, 'f': t_sFAST}
    # velocity: subwords
    CRAWL = 'c'
    SLOW = 's'
    MED = 'm'
    FAST = 'f'
    # velocity string to int dictionary
    vel_str2int = dict([(y,x) for x,y in enumerate(['c','s','m','f'])])


def getDistanceSubword(tickmag, slider=None):
    '''returns distance classification based on Word class'''
    if slider:  # different thresholds for slider speed
        if tickmag >= Word.t_sFAST:
            return Word.FAST
        elif tickmag >= Word.t_sMED:
            return Word.MED
        elif tickmag >= Word.t_sSLOW:
            return Word.SLOW
        else:
            return Word.CRAWL
    else:
        if tickmag >= Word.t_FAST:
            return Word.FAST
        elif tickmag >= Word.t_MED:
            return Word.MED
        elif tickmag >= Word.t_SLOW:
            return Word.SLOW
        else:
            return Word.CRAWL
    
    
def getDirectionSubword(v1, v2):
    '''
    returns cardinal direction of v2 given unit vectors v1 and v2
    assumes v1 is north
    '''
    # zero vector check
    if not np.any(v1) or not np.any(v2):
        return Word.N
    # difference between vectors
    rad_diff = np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0))
    hand_dir = np.cross(v1, v2)
    if math.isclose(hand_dir, 0.0):
        if rad_diff < Word.t_NNE:
            return Word.N
        else:
            return Word.S
    # right hand side
    if hand_dir < 0:
        if rad_diff <= Word.t_NNE:
            return Word.N
        elif rad_diff <= Word.t_NEE:
            return Word.NE
        elif rad_diff <= Word.t_ESE:
            return Word.E
        elif rad_diff <= Word.t_SES:
            return Word.SE
        else:
            return Word.S
    # left hand side
    else:
        if rad_diff <= Word.t_NNE:
            return Word.N
        elif rad_diff <= Word.t_NEE:
            return Word.NW
        elif rad_diff <= Word.t_ESE:
            return Word.W
        elif rad_diff <= Word.t_SES:
            return Word.SW
        else:
            return Word.S


def encodeTimePeriod(M, startTime, endTime, word):
    '''Encodes ticks within time period with word, returns last tick'''
    _,start_ti = M.getTick(startTime)
    _,end_ti = M.getTick(endTime)
    wordticks = list(range(start_ti,end_ti))
    if not wordticks:
        return end_ti
    for s_ti in wordticks:
        M.words[s_ti] = word
    return end_ti
    

def encodeSlider(M, obj, ti, T, direc):
    '''Encodes ticks along length of slider.'''
    c_type = obj['curveType']
    length = obj['length']
    ctrl_pts = []
    ctrl_pts.append([obj['x'], obj['y']])
    for d in obj['curvePoints']:
        ctrl_pts.append([d['x'], d['y']])
    # create curve object for calculating tick pos
    cv = Curve.from_kind_and_points(c_type,ctrl_pts,length)
    
    # get curve t idx corresponding to tick times
    ti_ms = T['beatLength'] / T['meter']
    _,ui_T = M.getUITDict(obj['time'])
    slide_ms, all_slide_ms = M.getSliderTimes(obj, T, ui_T)
    _,all_end_ti = M.getTick(obj['time']+all_slide_ms)
    
    # encode each slider tick beyond beginning tick
    prev_xy = [obj['x'], obj['y']]
    prev_direc = direc
    prev_ti = ti  # tick index
    all_slide_ti = list(range(ti+1,all_end_ti+1))
    for si, s_ti in enumerate(all_slide_ti):
        # ratio of one slide needed to get to this tick
        slide_ratio = (si+1) * ti_ms / slide_ms
        cv_t, rep = math.modf(slide_ratio)
        rep = int(rep)
        # get slider part using ratio
        if abs(round(cv_t)-cv_t) < 0.05:
            sw1 = Word.SLIDER_END
        else:  # middle of slider
            sw1 = Word.SLIDER_CENTER
        # use curve object to find tick position xy
        if rep % 2:  # odd, slider in reverse
            ti_xy = cv(1.0-cv_t)
        else:  # even
            ti_xy = cv(cv_t)
        # magnitude and direction of vector from prev tick xy
        mag, direc = getVector(prev_xy, ti_xy)
        # direction subword
        sw2 = getDirectionSubword(prev_direc, direc)
        # velocity subword only matters at slider ends
        if abs(round(cv_t)-cv_t) < 0.05:
            sw3 = getDistanceSubword(mag / 1, True)  # tickdiff in slider always 1
        else:  # middle of slider
            sw3 = Word.CRAWL  # subword won't be used during decoding
        
        # update for this tick
        M.words[s_ti] = sw1+'_'+sw2+'_'+sw3
        prev_xy = ti_xy
        prev_direc = direc
        prev_ti = s_ti
    
    return prev_xy, prev_direc, prev_ti
    
    
def encodeJSON2Words(M):
    '''
    Encodes given Map hitobjects into a sequence of "Word" strings
    Words give object type, direction, and velocity information for hitobjects
    '''
    # store previous obj information
    prev_xy = [DEFAULT_X, DEFAULT_Y]
    prev_direc = DEFAULT_DIREC
    prev_ti = -1  # tick index
    # initialize tick word array
    M.words = [None] * len(M.ticks)
    # deal with all hit objects
    for hi,obj in enumerate(M.O):
        
        obj_time = obj['time']
        # get object tick index given start time
        titime, ti = M.getTick(obj_time)
        tidiff = max(ti - prev_ti,1)  # min diff of 1 tick
        titimediff = np.abs(obj_time - titime)
        
        _,T = M.getTDict(obj_time)
        tilen = T['beatLength'] / T['meter']
        obj_xy = [obj['x'], obj['y']]
        # magnitude and direction of vector from prev object xy
        mag, direc = getVector(prev_xy, obj_xy)
        # direction subword
        sw2 = getDirectionSubword(prev_direc, direc)
        # velocity subword
        sw3 = getDistanceSubword(mag / tidiff)
        
        obj_class = obj['type']
        
        # hit circle
        if obj_class == 0:
            # check for half-tick hitcircle
            if titimediff > tilen*0.4:
                sw1 = Word.HITCIRCLE_DOUBLE
                # skip, only deal with objects on ticks
                continue
            else:
                sw1 = Word.HITCIRCLE
            M.words[ti] = sw1+'_'+sw2+'_'+sw3
        # slider
        elif obj_class == 1:
            # update with slider start tick
            sw1 = Word.SLIDER_BEGIN
            M.words[ti] = sw1+'_'+sw2+'_'+sw3
            # update rest of slider ticks and return slider end info
            obj_xy,direc,ti = encodeSlider(M, obj, ti, T, direc)
        # spinner
        elif obj_class == 2:
            end_time = obj['endTime']
            end_ti = encodeTimePeriod(M, obj_time,end_time,Word.SPINNER)
            # return spinner end tick info
            obj_xy,direc,ti = [DEFAULT_X,DEFAULT_Y], DEFAULT_DIREC, end_ti
        
        # update for calculating next obj tick info
        prev_xy = obj_xy
        prev_direc = direc
        prev_ti = ti
    
    # deal with breaks
    if M.E:
        for evt in M.E:
            encodeTimePeriod(M, evt['startTime'], evt['endTime'], Word.BREAK)
    # deal with empty ticks
    for ti, val in enumerate(M.words):
        if val == None:
            M.words[ti] = Word.EMPTY
    
    return M.words


# direction
def getNewDir(sw, prev_dir):
    '''Rotates previous direction by subword direction'''
    rot_vec = Word.rot[sw]
    cn, sn = rot_vec
    x,y = prev_dir
    new_dir = [0,0]
    new_dir[0] = x * cn - y * sn
    new_dir[1] = x * sn + y * cn
    return new_dir


def decodePos(prev_xy, prev_dir, tidiff, sw_dir, sw_vel, pix_ti=None):
    '''
    Finds new xy position using previous xy pos, # of ticks passed,
    and the direction and velocity subwords
    '''
    # magnitude
    if pix_ti:
        mag = pix_ti * tidiff
    else:
        if sw_vel == Word.CRAWL:
            dist_range = [Word.t_CRAWL, Word.t_SLOW-1]
        elif sw_vel == Word.SLOW:
            dist_range = [Word.t_SLOW, Word.t_MED-1]
        elif sw_vel == Word.MED:
            dist_range = [Word.t_MED, Word.t_FAST-1]
        else:
            dist_range = [Word.t_FAST, Word.t_FAST+100-1]
        
        # add randomness to map generation
        if tidiff > 1:
            mag = random.randrange(dist_range[0],dist_range[1]) * tidiff
        else:
            mag = (dist_range[0] + 20) * tidiff

    # check for large magnitude to prevent fruitless direction search
    dist = 240
    if mag > dist:
        xdist1, xdist2 = prev_xy[0], X_LIM[1]-prev_xy[0]
        ydist1, ydist2 = np.abs(prev_xy[1]), np.abs(Y_LIM[0] - prev_xy[1])
        if mag >= xdist1 and mag >= xdist2 and mag >= ydist1 and mag >= ydist2:
            mag = max(xdist1,xdist2,ydist1,ydist2) - 50
    
    # if new pos is out of bounds, find new direction
    prev_xy = np.array(prev_xy)
    new_dir = getNewDir(sw_dir, prev_dir)
    # add randomness to direction generation
    if tidiff > 1:
        x,y = new_dir
        randrad = random.uniform(0,np.pi/4)
        cn,sn = np.cos(randrad), np.sin(randrad)
        new_dir[0] = x * cn - y * sn
        new_dir[1] = x * sn + y * cn
    # check if new xy is out of bounds, if so search for other directions
    new_xy = prev_xy + np.array(new_dir) * mag
    dir_idx = 0
    rot_keys = list(Word.rot.keys())
    while not (X_LIM[0] <= new_xy[0] <= X_LIM[1]) or not (Y_LIM[0] <= new_xy[1] <= Y_LIM[1]):
        key = rot_keys[dir_idx]
        dir_idx += 1
        new_dir = getNewDir(key, prev_dir)
        new_xy = prev_xy + np.array(new_dir) * mag
        if dir_idx >= len(rot_keys) and (not (X_LIM[0] <= new_xy[0] <= X_LIM[1]) or not (Y_LIM[0] <= new_xy[1] <= Y_LIM[1])):
            pdb.set_trace()
    
    return list(new_xy), new_dir


def decodeTimePeriod(ti, words, w):
    '''
    Advances along [words] array from index [ti] until
    new word not matching [w] is reached
    Returns end tick index
    '''
    while ti < len(words):
        new_w = words[ti]
        if new_w != w:
            break
        ti += 1
    end_ti = ti - 1
    return end_ti


def decodeHitCircle(M, ti, words, prev_xy, prev_direc, prev_ti):
    '''
    Decodes hit circle word into map dictionary object
    Based on previous object's x-y position and direction
    Returns new x-y, direction
    '''
    w = words[ti]
    sw1, sw2, sw3 = w.split('_')
    start_time = M.ticks[ti]
    new_combo = int(ti % 10 == 0)  # renumber every X objects
    tidiff = ti - prev_ti
    new_xy, new_dir = decodePos(prev_xy, prev_direc, tidiff, sw2, sw3)
    obj = {'x': new_xy[0],
           'y': new_xy[1],
           'time': start_time,
           'hitsound': DEFAULT_HITSOUND,
           'newCombo': new_combo,
           'type': 0,
           'hitSample': DEFAULT_HITSAMPLE
           }
    M.O.append(obj)
    return new_xy, new_dir


def decodeSlider(M, ti, words, prev_xy, prev_dir, prev_ti):
    '''
    Decodes slider sequence of words into map json object
    Returns new tick index, x-y, direction of end of slider
    '''
    w = words[ti]
    sw1, sw2, sw3 = w.split('_')
    start_time = M.ticks[ti]
    new_combo = int(ti % 10 == 0)  # renumber every X objects
    tidiff = ti - prev_ti
    start_xy, start_dir = decodePos(prev_xy, prev_dir, tidiff, sw2, sw3)
    
    # get timing and slider speed info
    T_idx,T = M.getTDict(start_time)
    meter = T['meter']
    pixperti = (M.c_sx*T['speedMultiplier']*100) / meter
    if pixperti >= Word.t_sFAST:
        exp_sl_speed = Word.FAST
    elif pixperti >= Word.t_sMED:
        exp_sl_speed = Word.MED
    elif pixperti >= Word.t_sSLOW:
        exp_sl_speed = Word.SLOW
    else:
        exp_sl_speed = Word.CRAWL
    
    cv_pts = []
    repeats = 0
    total_len = 0
    prev_xy = start_xy
    prev_dir = start_dir
    # check next slider tick type
    next_w = words[ti + 1]
    # this slider ends at half tick and doesn't repeat
    if '_' not in next_w or (Word.SLIDER_CENTER not in next_w and Word.SLIDER_END not in next_w):
        next_pt, next_dir = decodePos(prev_xy,prev_dir,1,'N','c',pix_ti=pixperti/2)
        cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
        total_len = pixperti / 2
        new_xy, new_dir, end_ti = start_xy, next_dir, ti
    
    else:  # get info on rest of slider ticks
        ti_is_end = []        # keep track of which ticks are slider ends
        ti_dirs = []          # keep track of tick directions 
        num_N = 0             # for distinguishing half-tick len repeat sliders
        sl_speed = Word.SLOW  # keep track of slider velocity at slider end
        cur_ti = ti+1         # tick index iterator
        # as long as slider center or end, we know slider hasn't ended
        while Word.SLIDER_CENTER in next_w or Word.SLIDER_END in next_w:
            sw1,sw2,sw3 = next_w.split('_')
            if sw1 == Word.SLIDER_END:
                ti_is_end.append(1)
            else:
                ti_is_end.append(0)
            ti_dirs.append(sw2)
            if sw2 == Word.N:
                num_N += 1
            sl_speed = sw3
            cur_ti += 1
            next_w = words[cur_ti]
        
        # from slider speed judge whether to create new inherited timing section
        # don't do for small repeating sliders
        num_sl_ti = len(ti_is_end)
        if exp_sl_speed != sl_speed and sl_speed != Word.CRAWL:
            if sl_speed == Word.FAST:
                pixperti = Word.t_sFAST + 5
            elif sl_speed == Word.MED:
                pixperti = Word.t_sMED + 5
            elif sl_speed == Word.SLOW:
                pixperti = Word.t_sSLOW + 5
            else:
                pixperti = Word.t_sCRAWL + 5
            
            new_mult = pixperti*meter / (M.c_sx*100)
            new_blen = 100.0 / -new_mult
            T_obj = {'time': start_time,
                     'beatLength': new_blen,
                     'speedMultiplier': new_mult,
                     'meter': meter,
                     'timingChange': 0
                    }
            M.T_all.insert(T_idx+1,T_obj)  # create new time section to change slider speed
            M.T_sidx_all = np.insert(M.T_sidx_all,T_idx+1,start_time)  # update time sidx array
            
        # check for small, half-tick len repeat sliders
        if num_sl_ti > 1 and num_N == num_sl_ti and sum(ti_is_end) == num_sl_ti:
            next_pt, next_dir = decodePos(prev_xy,prev_dir,1,'N','c',pix_ti=pixperti/2)
            cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
            repeats = 2*num_sl_ti-1
            total_len = pixperti / 2
            new_xy, new_dir, end_ti = start_xy, next_dir, ti + num_sl_ti
        # get curve points by iterating through ticks
        else:
            end_ti = 0
            for cv_ti, ti_dir in enumerate(ti_dirs):
                next_pt, next_dir = decodePos(prev_xy,prev_dir,1,ti_dir,sl_speed,pix_ti=pixperti)
                cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
                # add duplicate point for multibezier
                if len(cv_pts) != 1 and len(cv_pts) % 4 == 0:
                    cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
                prev_xy, prev_dir = next_pt, next_dir
                # check for curve end
                if ti_is_end[cv_ti] == 1:
                    end_ti = cv_ti
                    break
            repeats = sum(ti_is_end)-1
            total_len = pixperti * (end_ti+1)
            new_xy, new_dir, end_ti = prev_xy, prev_dir, ti + num_sl_ti
    
    obj = {'x': start_xy[0],
           'y': start_xy[1],
           'time': start_time,
           'hitsound': DEFAULT_HITSOUND,
           'newCombo': new_combo,
           'type': 1,
           'curveType': 'B',
           'curvePoints': cv_pts,
           'repeatCount': repeats,
           'length': total_len,
           'edgeSounds': '|'.join([DEFAULT_EDGESOUND] * (2+repeats)),
           'edgeSets': '|'.join([DEFAULT_EDGESET] * (2 + repeats)),
           'hitSample': DEFAULT_HITSAMPLE
           }
    M.O.append(obj)
    return new_xy, new_dir, end_ti


def decodeSpinner(M, ti, words, prev_xy, prev_direc, prev_ti):
    '''
    Decodes spinner sequence of words into map json object
    Returns new tick index, x-y, and direction of end of slider
    '''
    w = words[ti]
    start_time = M.ticks[ti]
    # update ti to end of spinner object
    end_ti = decodeTimePeriod(ti,words,w)
    obj = {'x': DEFAULT_X,
            'y': DEFAULT_Y,
            'time': start_time,
            'hitsound': DEFAULT_HITSOUND,
            'newCombo': 1,  # always new combo on spinner
            'type': 2,
            'endTime': M.ticks[end_ti],
            'hitSample': DEFAULT_HITSAMPLE
            }
    M.O.append(obj)
    return [DEFAULT_X, DEFAULT_Y], DEFAULT_DIREC, end_ti


def decodeWords2JSON(M, words):
    '''
    Decodes given sequence of Word strings into python dictionary hit objects
    Requires Map object containing offset, tick, and timing info
    '''
    # make sure number of ticks is enough to hold all decoded words
    assert (len(words) <= len(M.ticks))
    
    # pandas makes working with strings a breeze
    words = pd.Series(words)
    
    # store previous obj information
    prev_xy = [DEFAULT_X, DEFAULT_Y]
    prev_direc = DEFAULT_DIREC
    prev_ti = -1  # tick index
    
    # iterate through words
    ti = 0
    while ti < len(words):
        w = words[ti]
        start_time = M.ticks[ti]
        if '_' in w:
            sw1, sw2, sw3 = w.split('_')
            if sw1 == Word.HITCIRCLE:
                obj_xy, direc = decodeHitCircle(M,ti,words,prev_xy,prev_direc,prev_ti)
            elif sw1 == Word.SLIDER_BEGIN:
                obj_xy, direc, ti = decodeSlider(M,ti,words,prev_xy,prev_direc,prev_ti)
            else:
                continue
            
            # update for calculating next obj tick info
            prev_xy = obj_xy
            prev_direc = direc
            prev_ti = ti
        
        elif w == Word.SPINNER:
            obj_xy, direc, ti = decodeSpinner(M,ti,words,prev_xy,prev_direc,prev_ti)
            
            # update for calculating next obj tick info
            prev_xy = obj_xy
            prev_direc = direc
            prev_ti = ti
        
        elif w == Word.BREAK:
            ti = decodeTimePeriod(ti,words,w)
            obj = {'startTime': start_time,
                   'endTime': M.ticks[ti]
                   }
            M.E.append(obj)
        
        ti += 1  # advance to next word


def decodeMap2Array(M):
    '''
    Decodes map and its sequence of Word strings into numpy array
    Requires Map object containing words, offset, ticks, and timing info
    This function returns 2 arrays
    One float arr for tick times and one int arr for words
    '''
    Mwords = M.words
    Mticks = M.ticks
    
    # tick times array
    # save on memory with single-precision float
    tick_arr = np.array(Mticks, dtype=np.float32)
    
    # words array
    # 3 columns, 1 - obj, 2 - direction, 3 - velocity
    obj_list, dir_list, vel_list = [],[],[]
    for w in Mwords:
        if '_' in w:
            sw1, sw2, sw3 = w.split('_')
            obj_list.append(sw1)
            dir_list.append(sw2)
            vel_list.append(sw3)
        else:
            obj_list.append(w)
            dir_list.append(Word.E)
            vel_list.append(Word.CRAWL)
    
    word_arr = np.zeros((len(Mticks),3), dtype=np.uint8)
    word_arr[:,0] = np.array([Word.obj_str2int[x] for x in obj_list], dtype=np.uint8)
    word_arr[:,1] = np.array([Word.dir_str2int[x] for x in dir_list], dtype=np.uint8)
    word_arr[:,2] = np.array([Word.vel_str2int[x] for x in vel_list], dtype=np.uint8)
    
    return tick_arr, word_arr




if __name__ == "__main__":
    # for testing only
    #filename = "umu. - humanly (Half) [Len's Robotically Another]"
    filename = "Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert]"
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    new_xy = decodePos([500, -380], [0,1], 1, 'SE', 'c')
    print(new_xy)
    pdb.set_trace()