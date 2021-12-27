# osu_seq2seq
Automatic sequence-to-sequence generation of beatmaps for the rhythm
game Osu!

Osu! is a popular free rhythm game where users submit "beatmaps"
that are mapped to a song's beat.
https://osu.ppy.sh/home

The classic game mode is similar to games such as Guitar Hero
or Beatsaber. An example of an impressive play can be seen here:
https://www.youtube.com/watch?v=AXOBMpSpsPU

Training a machine learning model for automatic beatmap generation
can be seen as learning a function for going from the audio
(music waveform) domain to the beatmap domain, where the
beatmap domain could be viewed as similar to music midi notes.
Thus this problem could be compared to the automatic music
transcription problem in literature.

_________________

Previously, github user kotritrona has done previous work on beatmap
prediction. A link to the repository is here:
https://github.com/kotritrona/osumapper

However, their model classifies one note at a time, which may not be
able to capture long-term patterns related to music sections
(intro, pre-chorus, chorus, etc) that can be seen in human-created
beatmaps.

Drawing inspiration from onsets and frames model: https://arxiv.org/pdf/1710.11153.pdf

Input data:
1. Training data
    Pre-timed music .wav file is downsampled to a constant mono channel 16 kHz sampling rate
    Songs are divided into 20 second sections with extra 0.032 ms before beginning of 20 sec section
	  Mel spectrograms extracted with 229 Mel bins, resulting in 626 time bins, 0.032 sec each
	
2. Training labels
    Ticks and objects from .osu maps translated into classes
    binary: 1 or 0 for presence of hitobject, multi: class for each of hitcircle, slider start, slider end
	  Each tick is mapped onto corresponding time bin within the 20.032 seconds/626 time bins

Preprocessing:
Each song file is pre-timed, meaning that each word is matched with a specific tick index in the song.
A "tick" is calculated from beat length divided by meter.
Beat length is calculated from (60*1000) / song BPM.
Meter is usually 4 ticks per beat (4/4 time in music theory).
Osu songs will range from 120 BPM to a high end of 400 BPM,
which corresponds to tick lengths in the range of 125 ms to 37.5 ms.
Time bin length of 32 ms for mel spectrograms is in range of highest BPM songs

In speech recognition and music classification literature, spectrograms have shown to be able to capture similar 
information by transforming audio data into frequency and time information.
CNNs also serve as excellent feature extractors on spectrograms. We can use a CNN network to summarize the 
information contained in a spectrogram that is generated from the music samples around each tick.




Thanks to Elessey for ranked beatmap dataset:
https://osu.ppy.sh/community/forums/topics/330552

Thanks to kotritrona for inspiration and previous work:
https://github.com/kotritrona/osumapper

