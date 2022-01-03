import pdb
import math
import random
import numpy as np

from math_tools import *


# set random seed for deterministic results
random.seed(2020)

# default center of osu play area
DEFAULT_X = 256
DEFAULT_Y = -192  # treating osu y axis as negative for my sanity
DEFAULT_DIR = np.array([0, 1])  # up direction (cardinal N, north)
DEFAULT_VEL = 0.0 # default velocity from last hitobject in osupix per second
DEFAULT_ANGDIF = 0.0 # default change in angle from last hitobject in radians

# osu play area limits
X_LIM = [0, 512]
Y_LIM = [-384, 0]

# always use normal hitsounds for now
DEFAULT_HITSOUND = 0
DEFAULT_EDGESOUND = "0"
DEFAULT_EDGESET = "0:0"
DEFAULT_HITSAMPLE = "0:0:0:0"


# word constants
# object subwords
HITCIRCLE = 'h'
SLIDER_BEGIN = 'slb'
SLIDER_CENTER = 'slc'
SLIDER_END = 'sle'
SPINNER = 'spin'
BREAK = 'b'
EMPTY = 'e'
# object string to int dictionary
obj_str2int = dict([(y,x) for x,y in enumerate(['e','b','spin','h','slb','slc','sle'])])
obj_int2str = {v: k for k,v in obj_str2int.items()}




# calculate slider osu!pixel per second from map inherited section:
# [inherited section speed mult] = 100.0 / - [inherited section beatLength property]
# [slider osu!pixel per beat] = ([Map slider mult]*[inherited section speed mult]*100)
# slider osu!pixel per second = [slider osu!pixel per beat] * (1000 / [beatLength (ms)])

# calculate osu!pixel per second (non-slider)
# [osupix per tick] = [hitobject osupix difference] / [hitobject tick difference]
# osupix per second = [osupix per tick] * (1000 / [tickLength (ms)])



def getDirectionDifference(v1, v2):
    '''
    Find difference between direction of vector 1 and vector 2 in radians
    '''
    # zero vector check
    if not np.any(v1):
        v1 = DEFAULT_DIR
    if not np.any(v2):
        return 0.0
    # difference between vectors
    rad_diff = np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0))
    hand_dir = np.cross(v1, v2).item()
    # either 0 or 180 degree difference, sign doesn't matter
    if math.isclose(hand_dir, 0.0):
        return rad_diff
    # return radian difference
    # negative is right (east) and positive is left (west)
    return rad_diff * np.sign(hand_dir)


def encodeTimePeriod(M, startTime, endTime, obj_type):
    '''
    Encodes ticks within time period with object type
    Encodes ticks with default velocity/direction (break or spinner)
    Returns ending tick
    '''
    _,start_ti = M.getTick(startTime)
    _,end_ti = M.getTick(endTime)
    obj_ti = list(range(start_ti,end_ti+1))
    if not obj_ti:
        return end_ti
    M.a_obj[obj_ti] = obj_str2int[obj_type]
    M.a_vel[obj_ti] = DEFAULT_VEL
    M.a_dir[obj_ti] = DEFAULT_ANGDIF
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
    _,ui_T = M.getUITDict(obj['time'])
    ti_ms = ui_T['beatLength'] / ui_T['meter']
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
        if abs(round(cv_t)-cv_t) < 0.05 or si == len(all_slide_ti)-1:
            sw1 = SLIDER_END
        else:  # middle of slider
            sw1 = SLIDER_CENTER
        # use curve object to find tick position xy
        if rep % 2:  # odd, slider in reverse
            ti_xy = cv(1.0-cv_t)
        else:  # even
            ti_xy = cv(cv_t)
        # magnitude and direction of vector from prev tick xy
        mag, direc = getVector(prev_xy, ti_xy)
        dir_change = getDirectionDifference(prev_direc, direc)
        pixperti = mag/1 # tickdiff in slider always 1
        pixpersec = pixperti * (1000/ti_ms)
        
        # update for this tick
        M.a_obj[s_ti] = obj_str2int[sw1]
        M.a_vel[s_ti] = pixpersec
        M.a_dir[s_ti] = dir_change
        prev_xy = ti_xy
        prev_direc = direc
        prev_ti = s_ti
    
    return prev_xy, prev_direc, prev_ti
    
    
def encodeMap2Array(M):
    '''
    Encodes given Map hitobjects into object arrays
    Arrays for hitobject type, velocity, and angular change (direction)
    Each array has length t, the number of ticks in the map timing
    '''
    # store previous obj information
    prev_xy = [DEFAULT_X, DEFAULT_Y]
    prev_direc = DEFAULT_DIR
    prev_ti = -1  # tick index
    # initialize hitobject and velocity/direction arrays
    t = len(M.ticks)
    M.a_obj = np.zeros((t, 1))
    M.a_vel = np.zeros((t, 1))
    M.a_dir = np.zeros((t, 1))
    
    # deal with all hit objects
    for hi,obj in enumerate(M.O):
        
        obj_time = obj['time']
        # get object tick index given start time
        titime, ti = M.getTick(obj_time)
        tidiff = max(ti - prev_ti,1)  # min diff of 1 tick
        titimediff = np.abs(obj_time - titime)
        
        _, uiT = M.getUITDict(obj_time)
        _, T = M.getTDict(obj_time)
        tilen = uiT['beatLength'] / uiT['meter']
        obj_xy = [obj['x'], obj['y']]
        # magnitude and direction of vector from prev object xy
        mag, direc = getVector(prev_xy, obj_xy)
        # direction difference between vectors
        dir_change = getDirectionDifference(prev_direc, direc)
        # velocity in osupix per second
        # osupix per second = [osupix per tick] * (1000 / [tickLength (ms)])
        pixperti = mag/tidiff
        pixpersec = pixperti * (1000/tilen)
        
        obj_class = obj['type']
        
        # hit circle
        if obj_class == 0:
            # check for half-tick hitcircle
            if titimediff > tilen*0.4:
                # skip, only deal with objects on ticks
                continue
            M.a_obj[ti] = obj_str2int[HITCIRCLE]
            M.a_vel[ti] = pixpersec
            M.a_dir[ti] = dir_change
        # slider
        elif obj_class == 1:
            # update with slider start tick
            M.a_obj[ti] = obj_str2int[SLIDER_BEGIN]
            M.a_vel[ti] = pixpersec
            M.a_dir[ti] = dir_change
            # update rest of slider ticks and return slider end info
            obj_xy,direc,ti = encodeSlider(M, obj, ti, T, direc)
        # spinner
        elif obj_class == 2:
            end_time = obj['endTime']
            end_ti = encodeTimePeriod(M, obj_time,end_time, SPINNER)
            # return spinner end tick info
            obj_xy,direc,ti = [DEFAULT_X,DEFAULT_Y], DEFAULT_DIR, end_ti
        
        # update for calculating next obj tick info
        prev_xy = obj_xy
        prev_direc = direc
        prev_ti = ti
    
    # deal with breaks
    if M.E:
        for evt in M.E:
            encodeTimePeriod(M, evt['startTime'], evt['endTime'], BREAK)
            
    return np.hstack((M.a_obj, M.a_vel, M.a_dir))


# direction
def getNewDir(dir, prev_dir):
    '''Rotates previous direction by radian direction'''
    cn, sn = np.cos(dir), np.sin(dir)
    x,y = prev_dir
    new_dir = [0,0]
    new_dir[0] = x * cn - y * sn
    new_dir[1] = x * sn + y * cn
    return new_dir


def decodePos(prev_xy, prev_dir, tidiff, dir, pix_ti=None):
    '''
    Finds new xy position using previous xy pos, # of ticks passed,
    and the direction and velocity values
    '''
    # magnitude (osupix difference between previous xy and new xy)
    mag = pix_ti * tidiff

    # check for large magnitude to prevent fruitless direction search
    dist = 240
    if mag > dist:
        xdist1, xdist2 = prev_xy[0], X_LIM[1]-prev_xy[0]
        ydist1, ydist2 = np.abs(prev_xy[1]), np.abs(Y_LIM[0] - prev_xy[1])
        if mag >= xdist1 and mag >= xdist2 and mag >= ydist1 and mag >= ydist2:
            mag = max(xdist1,xdist2,ydist1,ydist2) - 50
    
    # if new pos is out of bounds, find new direction
    prev_xy = np.array(prev_xy)
    new_dir = getNewDir(dir, prev_dir)
    
    # check if new xy is out of bounds, if so reflect
    new_xy = prev_xy + np.array(new_dir) * mag
    if new_xy[0] > X_LIM[1]:
        new_xy[0] = X_LIM[1] - (new_xy[0]-X_LIM[1])
    elif new_xy[0] < X_LIM[0]:
        new_xy[0] = X_LIM[0] + (X_LIM[0] - new_xy[0])
        
    if new_xy[1] > Y_LIM[1]:
        new_xy[1] = Y_LIM[1] - (new_xy[1]-Y_LIM[1])
    elif new_xy[1] < Y_LIM[0]:
        new_xy[1] = Y_LIM[0] + (Y_LIM[0]-new_xy[1])
    
    return list(new_xy), new_dir


def decodeTimePeriod(ti, arr, warr):
    '''
    Advances along array from index [ti] until
    new obj not matching [warr] is reached
    Returns end tick index
    '''
    while ti < arr.shape[0]:
        new_warr = arr[ti,:]
        if int(new_warr[0]) != int(warr[0]):
            break
        ti += 1
    end_ti = ti - 1
    return end_ti


def decodeHitCircle(M, ti, arr, prev_xy, prev_direc, prev_ti):
    '''
    Decodes hit circle array object into map dictionary object
    Based on previous object's x-y position and direction
    Returns new x-y, direction
    '''
    w = arr[ti]
    vel, dir = w[1], w[2]
    
    start_time = M.ticks[ti]
    _, T = M.getUITDict(start_time)
    ti_ms = T['beatLength'] / T['meter']
    new_combo = int(ti % 20 == 0)  # renumber every X ticks
    tidiff = ti - prev_ti
    # [osupix per tick] = [osupix per second] * ([tickLength (ms)] / 1000)
    pixperti = vel * (ti_ms / 1000)
    
    new_xy, new_dir = decodePos(prev_xy, prev_direc, tidiff, dir, pix_ti=pixperti)
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


def decodeSlider(M, ti, arr, prev_xy, prev_dir, prev_ti):
    '''
    Decodes slider sequence of array into map json object
    Returns new tick index, x-y, direction of end of slider
    '''
    w = arr[ti,:]
    vel, dir = w[1], w[2]
    
    start_time = M.ticks[ti]
    T_idx,T = M.getTDict(start_time)
    _, uiT = M.getUITDict(start_time)
    ti_ms = uiT['beatLength'] / uiT['meter']
    new_combo = int(ti % 20 == 0)  # renumber every X ticks
    tidiff = ti - prev_ti
    # [osupix per tick] = [osupix per second] * ([tickLength (ms)] / 1000)
    pixperti = vel * (ti_ms / 1000)
    start_xy, start_dir = decodePos(prev_xy, prev_dir, tidiff, dir, pix_ti=pixperti)
    
    # get timing and slider speed info
    meter = T['meter']
    pixperti = (M.c_sx*T['speedMultiplier']*100) / meter
    exp_sl_speed = pixperti
    
    cv_pts = []
    repeats = 0
    total_len = 0
    prev_xy = start_xy
    prev_dir = start_dir
    # check next slider tick type
    next_w = arr[ti+1, :]
    next_type = obj_int2str[int(next_w[0])]
    # this slider ends at half tick and doesn't repeat
    if next_type != SLIDER_CENTER and next_type != SLIDER_END:
        next_pt, next_dir = decodePos(prev_xy,prev_dir,1,DEFAULT_DIR,pix_ti=pixperti/2)
        cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
        total_len = pixperti / 2
        new_xy, new_dir, end_ti = start_xy, next_dir, ti
    
    else:  # get info on rest of slider ticks
        ti_is_end = []        # keep track of which ticks are slider ends
        ti_dirs = []          # keep track of tick directions 
        ti_vels = []          # keep track of tick velocities
        num_0 = 0             # for distinguishing half-tick len repeat sliders
        cur_ti = ti+1         # tick index iterator
        # as long as slider center or end, we know slider hasn't ended
        while next_type == SLIDER_CENTER or next_type == SLIDER_END:
            if next_type == SLIDER_END:
                ti_is_end.append(1)
            else:
                ti_is_end.append(0)
            ti_dirs.append(next_w[2])
            ti_vels.append(next_w[1] * (ti_ms / 1000))
            if math.isclose(next_w[2], 0.0):
                num_0 += 1
            cur_ti += 1
            next_w = arr[cur_ti, :]
            next_type = obj_int2str[int(next_w[0])]
        
        # from slider speed judge whether to create new inherited timing section
        # don't do for small repeating sliders
        num_sl_ti = len(ti_is_end)
        # convert slider pixpersecond to slider pixperti
        sl_speed = ti_vels[0]  # keep track of slider velocity at middle
        if not math.isclose(exp_sl_speed, sl_speed, rel_tol=0.05) and not math.isclose(sl_speed, 0.0):
            new_mult = sl_speed*meter / (M.c_sx*100)
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
        if math.isclose(sl_speed,0.0) and num_0 == num_sl_ti and sum(ti_is_end) == num_sl_ti:
            next_pt, next_dir = decodePos(prev_xy,prev_dir,1,0.0,pix_ti=pixperti/2)
            cv_pts.append({'x': next_pt[0], 'y': next_pt[1]})
            repeats = 2*num_sl_ti-1
            total_len = pixperti / 2
            new_xy, new_dir, end_ti = start_xy, next_dir, ti + num_sl_ti
        # get curve points by iterating through ticks
        else:
            end_ti = 0
            for cv_ti, ti_dir in enumerate(ti_dirs):
                sl_speed = ti_vels[cv_ti]
                next_pt, next_dir = decodePos(prev_xy,prev_dir,1,ti_dir,pix_ti=sl_speed)
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
            total_len = ti_vels[0] * (end_ti+1)
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


def decodeSpinner(M, ti, arr, prev_xy, prev_direc, prev_ti):
    '''
    Decodes spinner sequence of array objects into map json object
    Returns new tick index, x-y, and direction of end of slider
    '''
    w = arr[ti,:]
    start_time = M.ticks[ti]
    # update ti to end of spinner object
    end_ti = decodeTimePeriod(ti,arr,w)
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
    return [DEFAULT_X, DEFAULT_Y], DEFAULT_DIR, end_ti


def decodeArray2Map(M, arr):
    '''
    Decodes given arrays into python dictionary hit objects
    [arr]: 3-col array, each col represents object, velocity, and direction arrays
    Requires Map object containing correct offset, tick, and timing info
    '''
    # make sure number of ticks is enough to hold all decoded words
    assert (arr.shape[0] <= len(M.ticks))
    
    # store previous obj information
    prev_xy = [DEFAULT_X, DEFAULT_Y]
    prev_direc = DEFAULT_DIR
    prev_ti = -1  # tick index
    
    # iterate through each tick object
    ti = 0
    while ti < arr.shape[0]:
        warr = arr[ti,:]
        start_time = M.ticks[ti]
        obj_typeint = warr[0].astype(np.int)
        obj_type = obj_int2str[obj_typeint]
        
        if obj_type == BREAK:
            ti = decodeTimePeriod(ti,arr,warr)
            obj = {'startTime': start_time,
                   'endTime': M.ticks[ti]
                   }
            M.E.append(obj)
        
        else:
            if obj_type == HITCIRCLE:
                obj_xy, direc = decodeHitCircle(M,ti,arr,prev_xy,prev_direc,prev_ti)
                
            elif obj_type == SLIDER_BEGIN:
                obj_xy, direc, ti = decodeSlider(M,ti,arr,prev_xy,prev_direc,prev_ti)
                
            elif obj_type == SPINNER:
                obj_xy, direc, ti = decodeSpinner(M,ti,arr,prev_xy,prev_direc,prev_ti)
            
            else:  # empty
                ti += 1
                continue
            
            # update for calculating next obj tick info
            prev_xy = obj_xy
            prev_direc = direc
            prev_ti = ti
        
        ti += 1  # advance to next word




if __name__ == "__main__":
    # for testing only
    import matplotlib.pyplot as plt
    
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    #filename = "Yuyoyuppe - AiAe (Fort) [Eternal]"
    filename = "YOASOBI - Ano Yume o Nazotte (Sarawatlism) [Daisuki]"
    #time_bpm = [[-30,200,4]]
    #time_bpm = [[1008, 180, 4]]
    time_bpm = [[1342,180,4]]
    
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    
    from map import Map
    m = Map.fromPath(osu_file)
    # encode then decode the hitobjects and try out the map
    obj_arr = m.encodeTicks()
    plt.plot(obj_arr[:,0])
    plt.show(block=False)
    pdb.set_trace()
    
    # new map from timing only
    import librosa
    sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000
    m_empty = Map.fromTiming(time_bpm,mp3_file,mp3_len=mp3_len)
    
    m_empty.decodeArray(obj_arr)
    m_empty.saveMap2Osu()
    pdb.set_trace()