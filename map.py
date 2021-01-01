
import os
import re
import sys
import pdb
import ntpath
import numpy as np
import pandas as pd

from map_io import load_map, load_template, save_map, save_words
from word import *
from utils import parseOsuMp3Filename


class Map:
    '''map object class for reading, writing, and analyzing osu map info'''

    def __init__(self, name, map_json, mp3_len=None):
        
        if mp3_len:
            self.mp3_len = mp3_len
        
        self.name = name
        
        # json style dicts
        self.version = map_json['Version']
        self.G = map_json['General']
        self.Ed = map_json['Editor']
        self.M = map_json['Metadata']
        self.D = map_json['Difficulty']
        self.E = map_json['Events']
        self.T = map_json['TimingPoints']
        self.C = map_json['Colours']
        self.O = map_json['HitObjects']

        self.mode = self.G['Mode']
        
        # constants
        self.c_HP = self.D['HPDrainRate']
        self.c_CS = self.D['CircleSize']
        self.c_OD = self.D['OverallDifficulty']
        self.c_AR = self.D['ApproachRate']
        
        self.c_sx = self.D['SliderMultiplier']
        self.c_ln = self.G['AudioLeadIn']
        self.c_nO = len(self.O)
        
        # timing info
        self.timeSections()
    

    @classmethod
    def fromPath(cls,osu_path):
        '''create map using osu file'''
        # error checking
        #if not os.path.isfile(mp3_path):
        #    raise ValueError('mp3 file not found')
        if not os.path.isfile(osu_path):
            raise ValueError('osu file not found')
        #if os.path.splitext(mp3_path)[1] != '.mp3':
        #    raise ValueError('not mp3 file')

        # load .osu map info
        map_json = None
        try:
            map_json = load_map(osu_path,False)
        except:
            print("error loading .osu file: ", sys.exc_info()[0])
            return None
        
        if map_json is None:
            return None
        # skip other game modes for now
        if map_json['General']['Mode'] != 0:
            return None
        # use basename of osu file as id for map
        name = ntpath.basename(osu_path)

        return cls(name,map_json)
    

    @classmethod
    def fromTiming(cls,time_bpm,mp3_path,mp3_len):
        '''
        create map object using timing info and mp3 file
        time_bpm => (list) each element is a 3 element list: [time, bpm, meter] 
                    indicating time, bpm, and meter of uninherited timing sections
                    time (int), bpm (float), meter (int)
        mp3_path => (str) path to mp3 file for this map
        mp3_len  => (float) mp3 file length in milliseconds
        '''
        # use basename of mp3 file as id for map
        name = ntpath.splitext(ntpath.basename(mp3_path))[0]
        artist, song, _ = parseOsuMp3Filename(name)
        # edit difficulty name to be "[diff name]"
        diff = 'AIs Easy'
        if '[' in name:
            name = re.sub('\[.*\]','['+diff+']',name)
        else:
            name = name + ' [' + diff + ']'
        
        # get template map json
        map_json = load_template(name, artist, song, diff)
        
        # populate timing dictionary
        for ui_info in time_bpm:
            ui = {'time': ui_info[0],
                  'beatLength': (60*1000) / ui_info[1],
                  'bpm': ui_info[1],
                  'speedMultiplier': 1.0,
                  'meter': ui_info[2],
                  'timingChange': 1
                 }
            map_json['TimingPoints'].append(ui)

        return cls(name,map_json,mp3_len)
    
    
    def saveMap2Osu(self):
        '''Write stored map hit object dictionary to .osu file'''
        # json style dicts
        map_json = {'Version': self.version,
                    'General': self.G,
                    'Editor': self.Ed,
                    'Metadata': self.M,
                    'Difficulty': self.D,
                    'Events': self.E,
                    'TimingPoints': self.T_all,
                    'Colours': self.C,
                    'HitObjects': self.O}
        
        save_map(self.name, map_json)
    
    
    def saveWords2JSON(self, json_path):
        '''Saves difficulty, timing info, and word list into json'''
        # get uninherited timing section data
        time_bpm = []
        for ui in self.T_ui:
            begin_time = ui["time"]
            end_time = ui['endTime']
            bpm = ui["bpm"]
            mspb = ui["beatLength"]
            mspti = mspb / ui['meter']
            meter = ui['meter']
            time_bpm.append({'time': begin_time,
                             'end_time': end_time,
                             'bpm': bpm,
                             'mspb': mspb,
                             'mspti': mspti,
                             'meter': meter
                             })
        out = {}
        out['difficulty'] = {'HP': self.c_HP,
                             'CS': self.c_CS,
                             'OD': self.c_OD,
                             'AR': self.c_AR
                             }
        out['time_bpm'] = time_bpm
        out['ticks'] = list(self.ticks)
        out['words'] = self.words
        save_words(json_path,out)
    
    
    def getTick(self, timeidx):
        '''Get current tick index from time'''
        timediff = np.abs(self.ticks - timeidx)
        closest_ti = timediff.argmin()
        return self.ticks[closest_ti], closest_ti
        
        
    def getTDict(self, timeidx):
        '''Get the dict info of time section [timeidx] belongs to'''
        T_idx = len(self.T_sidx_all[self.T_sidx_all <= timeidx]) - 1
        return T_idx, self.T_all[T_idx]
    
    
    def getUITDict(self, timeidx):
        '''Get the uninherited time section [timeidx] belongs to'''
        T_idx = len(self.T_sidx_ui[self.T_sidx_ui <= timeidx]) - 1
        return T_idx, self.T_ui[T_idx]
    
    
    def getSliderTimes(self, sliderobj, cur_T, ui_T):
        '''Get time for one slider length as well as entire slider (with repeats)'''
        blen = cur_T['beatLength']
        sx_mult = cur_T['speedMultiplier']
        # if bpm was doubled make sure to double slider tick lengths as well
        length = sliderobj['length'] * ui_T['sliderLengthMult']
        slide_time = length / (self.c_sx*sx_mult*100) * blen
        total_slide_time = slide_time * (sliderobj['repeatCount'] + 1)
        return slide_time, total_slide_time


    def timeSections(self):
        '''Setup timing sections and ticks (1/[meter] of a beat, typically 1/4)'''
        time_list = self.T
        # get offset from first uninherited timing section
        first_ui = self.T[0]
        offset = first_ui['time']
        beat_len = first_ui['beatLength']
        # if offset is negative set to positive
        while offset < 0:
            offset += beat_len
        self.T[0]['time'] = offset
        self.offset = offset
        
        # uninherited and all timing sections
        self.T_ui = []
        self.T_all = []
        # save starting points of every ui and timing section for lookup
        self.T_sidx_ui = []
        self.T_sidx_all = []
        cur_beatLen = 0.0
        for sec in time_list:
            if sec['timingChange'] == 1:
                bpm = 1 / sec['beatLength'] * 60 * 1000
                # double bpm if below 120
                if bpm < 120:
                    sec['beatLength'] = sec['beatLength'] / 2
                    bpm *= 2
                    sec['sliderLengthMult'] = 2
                else:
                    sec['sliderLengthMult'] = 1
                sec['bpm'] = bpm
                cur_beatLen = sec['beatLength']
                self.T_ui.append(sec)
                self.T_sidx_ui.append(sec['time'])
            else:
                sec['speedMultiplier'] = 100.0 / -float(sec['beatLength'])
                sec['beatLength'] = cur_beatLen
                sec['bpm'] = 1 / float(cur_beatLen) * 1000 * 60
            self.T_all.append(sec)
            self.T_sidx_all.append(sec['time'])
        # convert to numpy for easier lookup operations
        self.T_sidx_ui = np.array(self.T_sidx_ui)
        self.T_sidx_all = np.array(self.T_sidx_all)
        
        # get timing ticks
        self.ticks = np.array([])
        for i, ui in enumerate(self.T_ui):
            begin_time = ui["time"]
            mspb = ui["beatLength"]
            if i < len(self.T_ui)-1:
                end_time = self.T_ui[i+1]["time"]
            elif self.O:
                lastobj = self.O[-1]
                lastpad = 1000 # add 1s after the last note
                length = lastobj["time"] + lastpad
                if lastobj["type"] == 1: # slider
                    # lookup timing section idx
                    _, cur_T = self.getTDict(lastobj['time'])
                    _, ui_T = self.getUITDict(lastobj['time'])
                    _, all_slide_time = self.getSliderTimes(lastobj, cur_T, ui_T)
                    # accounts for slider end time
                    length = lastobj['time'] + all_slide_time + lastpad
                if lastobj["type"] == 2: # spinner end
                    length = lastobj["endTime"] + lastpad
                end_time = length
            else:  # no hitobjects, use mp3 length
                end_time = self.mp3_len
            
            msp4b = mspb / ui['meter']
            #endgap = (end_time - begin_time) % msp4b
            arr = np.arange(begin_time, end_time, msp4b)
            # save ending time of each ui section
            self.T_ui[i]['endTime'] = end_time#-endgap
            self.ticks = np.append(self.ticks,arr)
        
        
    def encodeTicks(self):
        '''
        Encodes each tick (quarter beat) into a custom "word"
        Words give object type, direction, and velocity information for hitobjects
        '''
        return encodeJSON2Words(self)
    
    
    def decodeWords(self,words):
        '''Decodes words and saves hit object json to Map object'''
        decodeWords2JSON(self,words)
        self.words = words


def profile_map():
    # for testing only
    #filename = "Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity]"
    #filename = "Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert]"
    filename = "YOASOBI - Ano Yume o Nazotte (Sarawatlism) [Daisuki]"
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    #filename = "xi - FREEDOM DiVE (Pikastar) [Universe]"
    
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    
    # [Time(ms), bpm, meter(beats per measure)]
    # assume timing is found beforehand
    #time_bpm = [[15688, 175, 4]]
    #time_bpm = [[540,91,4],[2245,89,4]]
    time_bpm = [[1342,180,4]]
    #time_bpm = [[-30,200,4]]
    #time_bpm = [[2133,222.22,4]]
    
    import librosa
    sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000
    m_empty = Map.fromTiming(time_bpm,mp3_file,mp3_len=mp3_len)
    
    # read actual .osu map
    m = Map.fromPath(osu_file)
    # encode then decode the hitobjects and try out the map
    obj_words = m.encodeTicks()
    m.saveWords2JSON("songs/test.json")
    
    #m_empty.decodeWords(obj_words)
    #m_empty.saveMap2Osu()


if __name__ == "__main__":
    
    profile_map()