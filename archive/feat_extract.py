
import os
import sys
from random import shuffle
import numpy as np
from scipy.io import wavfile
import yaml
import librosa

from tqdm import tqdm


if __name__=='__main__':
    
    import librosa.display
    import pdb
    import time
    import matplotlib.pyplot as plt
    
    # don't use librosa for reading, use scipy
    song_file = "songs/osu_mp3/Our Stolen Theory - United (L.A.O.S Remix) (Asphyxia) [Infinity].wav"
    sr,x = wavfile.read(song_file)
    
    x = x.astype(np.float)
    # normalize audio
    x = (x - np.mean(x)) / np.std(x)
    num_s = x.shape[0]/sr
    
    """ # mel spec
    w_stride = (1024,512)
    fft_size = 1024
    n_filts = 128
    
    st = time.time()
    n_chunks = 2
    chunk = round(np.floor(x.shape[0]/n_chunks))
    for i in range(n_chunks):
        xtemp = x[i*chunk:(i+1)*chunk]
        spec = librosa.feature.melspectrogram(xtemp,sr,n_fft=fft_size,hop_length=w_stride[1],win_length=w_stride[0])
        #spec = mel_spec(xtemp,sr,window_stride=w_stride,fft_size=fft_size,num_filt=n_filts)
    
    et = time.time() - st
    print("mel took ", et, " sec with chunk size ", chunk/sr, " sec")
    
    #spec_db = librosa.power_to_db(spec,ref=np.max)
    plt.figure(figsize=(15,5))
    librosa.display.specshow(spec,x_axis='time')
    plt.colorbar()
    plt.show()
    
    pdb.set_trace() """
    
    # cqt
    bin_per_oct = 12
    num_oct = 8
    n_bins = bin_per_oct * num_oct
    
    hop_len = 2**8  # smallest hop length possible
    fmin = 27.5  # lowest piano key is 32.703 Hz
    
    # with 8 octaves, our freq range is [27.5, 6644.875]
    #cqt_range = librosa.core.cqt_frequencies(n_bins,fmin,bin_per_oct)
    
    # time spectrogram generation
    n_chunks = 1
    chunk = int(round(np.floor(x.shape[0]/n_chunks)))
    st = time.time()
    for i in range(n_chunks):
        pdb.set_trace()
        xtemp = x[i*chunk:(i+1)*chunk]
        spec = librosa.cqt(xtemp,sr=sr,fmin=fmin,bins_per_octave=bin_per_oct,n_bins=n_bins,hop_length=hop_len)
    
    et = time.time() - st
    print("cqt took ", et, " sec with chunk size ", chunk/sr, " sec")
    
    # split spectrogram into strided windows around each tick
    logSpec = librosa.amplitude_to_db(np.abs(spec))
    pdb.set_trace()
    plt.figure(figsize=(15,5))
    librosa.display.specshow(logSpec, sr=sr, x_axis='time',y_axis='cqt_hz', hop_length=hop_len,fmin=fmin,bins_per_octave=bin_per_oct,cmap='coolwarm')
    plt.show()
    
    pdb.set_trace()