# -*- coding: utf-8 -*-

#
# Read .osu file as json
# similar to load_map.js
# modified from https://gist.github.com/marios8543/71e559f575b72088eaf0cc6495bfa483
#
# .osu file format expected to be v14

import re
import os
import json
import pdb

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def get_line(phrase,osu):
    for num, line in enumerate(osu, 0):
        if phrase in line:
            return num


def add_list(osu_list,out_obj):
    for item in osu_list:
        if ':' in item:
            item = [a.strip() for a in item.split(':')]
            if isfloat(item[1]):
                out_obj[item[0]] = float(item[1])
            else:
                out_obj[item[0]] = item[1]


def add_events_list(osu_list,out_obj):
    for item in osu_list:
        if ',' in item:
            item = [a.strip() for a in item.split(',')]
            if item[0] == '2' or item[0] == 'Break':
                point = {
                    'startTime': int(item[1]),
                    'endTime': int(item[2])
                }
                out_obj.append(point)


def add_timing_list(osu_list,out_obj):
    for item in osu_list:
        if ',' in item:
            item = [a.strip() for a in item.split(',')]
            try:
                time = int(item[0])
                point = {
                    'time': time,
                    'beatLength': float(item[1]),
                    'bpm': 1 / float(item[1]) * 1000 * 60,
                    'speedMultiplier': (100.0 / -float(item[1])) if float(item[1])< 0 else 1.0
                }
                if len(item) >= 3:
                    point['meter'] = int(item[2])
                if len(item) >= 7:
                    point['timingChange'] = int(item[6])
                out_obj.append(point)
                
            except ValueError:
                continue
                


def add_hitobject_list(osu_list,out_obj):
    for item in osu_list:
        if ',' in item:
            item = [a.strip() for a in item.split(',')]
            type = int(item[3])
            point = {
                'x': int(item[0]),
                'y': -int(item[1]),  # osu coord easier if y neg
                'time': int(item[2]),
                'hitsound': int(item[4])
            }
            # new combo
            if type & 0b100 == 0b100:
                point['newCombo'] = 1
            else:
                point['newCombo'] = 0
                
            # hit circle
            if type & 0b1 == 0b1:
                point['type'] = 0
                if len(item) >= 6:
                    point['hitSample'] = item[5]
                    
            # slider
            elif type & 0b10 == 0b10:
                point['type'] = 1
                curveInfo = [a.strip() for a in item[5].split('|')]
                point['curveType'] = curveInfo[0]
                point['curvePoints'] = []
                for pt in curveInfo[1:]:
                    ptxy = pt.split(':')
                    pt_dict = {
                        'x': int(ptxy[0]),
                        'y': -int(ptxy[1])
                    }
                    point['curvePoints'].append(pt_dict)
                point['repeatCount'] = max(0,int(item[6])-1)
                point['length'] = float(item[7])
                
                if len(item) >= 9:
                    point['edgeSounds'] = item[8]
                if len(item) >= 10:
                    point['edgeSets'] = item[9]
                if len(item) >= 11:
                    point['hitSample'] = item[10]
                    
            # spinner
            elif type & 0b1000 == 0b1000:
                point['type'] = 2
                point['endTime'] = int(item[5])
                if len(item) >= 7:
                    point['hitSample'] = item[6]
                    
            # osu!mania hold
            elif type & 64 == 64:
                point['type'] = 3
                point['endTime'] = int(item[5])
                if len(item) >= 7:
                    point['hitSample'] = item[6]
            
            out_obj.append(point)


def load_template(mp3_name,artist_name, song_name, diff):
    '''load template map json dictionary using mp3 filename'''
    
    out = {}
    out['Version'] = 14
    out['General'] = {'AudioFilename': mp3_name+'.mp3',
                      'AudioLeadIn': 0,
                      'PreviewTime': 10000,  # arbitrary preview
                      'Countdown': 0,
                      'SampleSet': "Normal",
                      'StackLeniency': 0.0,
                      'Mode': 0,
                      'LetterboxInBreaks': 0,
                      'EpilepsyWarning': 0,
                      'WidescreenStoryBoard': 0
                      }
    out['Editor'] = {'Bookmarks': 10000,
                      'DistanceSpacing': 1.0,
                      'BeatDivisor': 4,
                      'GridSize': 32,
                      'TimelineZoom': 3
                    }
    non_unicode_name = song_name.encode('ascii','ignore')
    non_unicode_name = non_unicode_name.decode()
    non_unicode_artist = artist_name.encode('ascii','ignore')
    non_unicode_artist = non_unicode_artist.decode()
    out['Metadata'] = {'Title': non_unicode_name,
                       'TitleUnicode': song_name,
                       'Artist': non_unicode_artist,
                       'ArtistUnicode': artist_name,
                       'Creator': "Trackest",
                       'Version': diff,
                       'Source': "",
                       'Tags': "AI_generated"
                      }
    # default difficulty settings
    out['Difficulty'] = {'HPDrainRate': 5,
                         'CircleSize': 4,
                         'OverallDifficulty': 7,
                         'ApproachRate': 9,
                         'SliderMultiplier': 1.4,
                         'SliderTickRate': 1
                        }
    out['Events'] = []
    out['TimingPoints'] = []
    out['Colours'] = {'Combo1': '247,157,157',
                      'Combo2': '224,177,252',
                      'Combo3': '251,217,153'
                      }
    out['HitObjects'] = []
    return out


def load_map(file,save_f):

    osu = open(file,encoding="utf-8").readlines()
    
    # delete comments
    for lnum, line in enumerate(osu):
        if '//' in line:
            subline = line.split("//")
            osu[lnum] = subline[0]
    
    version_line = get_line('osu file format v',osu)
    
    out = {}
    out['Version'] = int(''.join(re.findall(r'\d+',osu[version_line])))
    out['General'] = {}
    out['Editor'] = {}
    out['Metadata'] = {}
    out['Difficulty'] = {}
    out['Events'] = []
    out['TimingPoints'] = []
    out['Colours'] = {}
    out['HitObjects'] = []
    
    general_line = get_line('[General]',osu)
    editor_line = get_line('[Editor]',osu)
    metadata_line = get_line('[Metadata]',osu)
    difficulty_line = get_line('[Difficulty]',osu)
    events_line = get_line('[Events]',osu)
    timing_line = get_line('[TimingPoints]',osu)
    colour_line = get_line('[Colours]',osu)
    hit_line = get_line('[HitObjects]',osu)
    
    general_list = osu[general_line:editor_line-1]
    editor_list = osu[editor_line:metadata_line-1]
    metadata_list = osu[metadata_line:difficulty_line-1]
    difficulty_list = osu[difficulty_line:events_line-1]
    events_list = osu[events_line:timing_line-1]
    timingpoints_list = osu[timing_line:colour_line-1]
    colour_list = osu[colour_line:hit_line-1]
    hitobject_list = osu[hit_line:]
    
    add_list(general_list,out['General'])
    add_list(editor_list,out['Editor'])
    add_list(metadata_list,out['Metadata'])
    add_list(difficulty_list,out['Difficulty'])
    add_events_list(events_list,out['Events'])
    add_timing_list(timingpoints_list,out['TimingPoints'])
    add_list(colour_list,out['Colours'])
    add_hitobject_list(hitobject_list,out['HitObjects'])
    
    output = json.dumps(out, indent=4)#.replace('\n','')
    orig_file, orig_ext = os.path.splitext(file)
    outdir, outname = os.path.split(orig_file)
    outdir = os.path.dirname(outdir)
    outfile = os.path.join(outdir,outname+'.json')
    
    if save_f:
        with open(outfile,'w') as file:
            file.write(output)
        print(outfile+' written successfully')
    
    return out


def save_map(basename, map_json):
    savepath = 'songs/'
    osu_path = savepath + basename + '.osu'
    
    def writeDict(f,d):
        for k,v in d.items():
            f.write(k + ': ' + str(v) + '\n')
    
    def writeEvents(f,l):
        f.write('//Break Periods\n')
        for br in l:
            f.write('2,'+str(int(br['startTime']))+','+str(int(br['endTime']))+'\n')
    
    def writeTimingPoints(f,l):
        SAMPLESET = 0
        SAMPLEIDX = 0
        EFFECTS = 0
        VOLUME = 80
        for tp in l:
            out_str = '{:n},{:f},{:n},{:n},{:n},{:n},{:n},{:n}\n'.format(
                round(tp['time']),tp['beatLength'],tp['meter'],SAMPLESET,SAMPLEIDX,
                VOLUME,tp['timingChange'],EFFECTS)
            f.write(out_str)
    
    def writeHitObj(f,l):
        HITSOUND = 0
        for obj in l:
            x,y,time = round(obj['x']), round(-obj['y']), round(obj['time'])
            obj_type = obj['type']
            new_combo = obj['newCombo']
            hitsample = obj['hitSample']
            if new_combo == 1:
                combo_bin = 0b100
            else:
                combo_bin = 0b000
            
            if obj_type == 0:
                bin_type = 0b1 | combo_bin
                out_str = '{:n},{:n},{:n},{:n},{:n},{}\n'.format(
                    x,y,time,bin_type,HITSOUND,hitsample)
            elif obj_type == 1:
                bin_type = 0b10 | combo_bin
                cv_type = obj['curveType']
                cv_pts = []
                cv_pts.append(cv_type)
                for cv_pt in obj['curvePoints']:
                    cv_pts.append(str(round(cv_pt['x']))+':'+str(round(-cv_pt['y'])))
                cv_str = '|'.join(cv_pts)
                out_str = '{:n},{:n},{:n},{:n},{:n},{},{:n},{:n},{},{},{}\n'.format(
                    x,y,time,bin_type,HITSOUND,cv_str,obj['repeatCount']+1,
                    round(obj['length']),obj['edgeSounds'],obj['edgeSets'],obj['hitSample'])
            elif obj_type == 2:
                bin_type = 0b1000 | combo_bin
                out_str = '{:n},{:n},{:n},{:n},{:n},{:n},{}\n'.format(
                    x,y,time,bin_type,HITSOUND,round(obj['endTime']),hitsample)
            
            f.write(out_str)
    
    with open(osu_path,'w') as file:
        file.write('osu file format v'+str(map_json['Version'])+'\n\n')
        file.write('[General]\n')
        writeDict(file,map_json['General'])
        file.write('\n')
        file.write('[Editor]\n')
        writeDict(file,map_json['Editor'])
        file.write('\n')
        file.write('[Metadata]\n')
        writeDict(file,map_json['Metadata'])
        file.write('\n')
        file.write('[Difficulty]\n')
        writeDict(file,map_json['Difficulty'])
        file.write('\n')
        file.write('[Events]\n')
        writeEvents(file,map_json['Events'])
        file.write('\n')
        file.write('[TimingPoints]\n')
        writeTimingPoints(file,map_json['TimingPoints'])
        file.write('\n')
        file.write('[Colours]\n')
        writeDict(file,map_json['Colours'])
        file.write('\n')
        file.write('[HitObjects]\n')
        writeHitObj(file,map_json['HitObjects'])
    
    print(osu_path + ' written successfully')


def save_words(json_path,out_dict):
    '''Save word and timing lists to json'''
    output = json.dumps(out_dict, indent=4)
    with open(json_path,'w') as file:
        file.write(output)
    print(json_path + ' written successfully')


if __name__ == "__main__":
    file = "songs/osu_mp3/Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity].osu"
    m = load_map(file,True)