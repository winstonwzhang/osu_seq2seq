
import numpy as np
from scipy.io import wavfile
import librosa



def get_spec(wav_f, config):
    '''Using librosa'''
    nfft = config.feature.nfft
    hop = config.feature.hop
    n_mels = config.feature.n_mels
    fmin = config.feature.fmin
    
    sr,x = wavfile.read(wav_f)
    x = x.astype(np.float)
    # get wav time
    song_len = len(x)/sr
    #print('song length: ', song_len, ' seconds')
    # crop audio to be multiple of hop length (512)
    new_len = int(np.floor(len(x) / hop) * hop)
    #print('cropped length: ', new_len / sr, ' seconds')
    x = x[:new_len]
    # normalize audio
    norm_x = librosa.util.normalize(x)

    st = time.time()
    # fft time bins 2048 samples (0.128 sec) long with 512 sample hop (0.032 sec)
    # f_max is default librosa value according to onsets and frames
    spec = librosa.feature.melspectrogram(norm_x, sr, n_fft=nfft, 
                                          hop_length=hop, n_mels=n_mels, fmin=fmin)#, htk=True)
    spec = librosa.power_to_db(spec)
    et = time.time() - st
    print("mel took ", et, " sec")
    return spec, x, sr, new_len/sr


if __name__=='__main__':
    # for testing only
    import matplotlib.pyplot as plt
    
    # feature extraction parameters
    configfile = 'config/params2021.yaml'
    from util import AttrDict
    import yaml
    with open(configfile) as param_f:
        config = AttrDict(yaml.safe_load(param_f))
    
    #filename = "Caravan Palace - Miracle (Mulciber) [Extra]"
    filename = "Yuyoyuppe - AiAe (Fort) [Eternal]"
    #time_bpm = [[-30,200,4]]
    time_bpm = [[1008, 180, 4]]
    
    osu_file = "songs/osu_mp3/" + filename + ".osu"
    mp3_file = "songs/osu_mp3/" + filename + ".mp3"
    
    from map import Map
    m = Map.fromPath(osu_file)
    # encode then decode the hitobjects and try out the map
    obj_arr = m.encodeTicks()
    
    # new map from timing only
    sec_len = librosa.get_duration(filename=mp3_file)
    mp3_len = sec_len * 1000
    m_empty = Map.fromTiming(time_bpm,mp3_file,mp3_len=mp3_len)
    
    m_empty.decodeArray(obj_arr)
    m_empty.saveMap2Osu()
    pdb.set_trace()