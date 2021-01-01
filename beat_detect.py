
import numpy as np
import matplotlib.pyplot as plt
import madmom.features.beats as mmb


if __name__=='__main__':
    # get song activations
    song_file = "songs/osu_mp3/Will Stetson - Despacito ft. R3 Music Box (Sotarks) [Monstrata's Slow Expert].mp3"
    act = mmb.RNNBeatProcessor()(song_file)
    
    # use DBN beat tracker
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