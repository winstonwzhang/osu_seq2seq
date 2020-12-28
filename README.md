# osu_seq2seq
Automatic sequence-to-sequence generation of beatmaps for the rhythm
game Osu! using NLP concepts

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
Thus this problem is could be compared to the automatic music
transcription problem in literature.

However, besides the goal of classifying note class and note
duration, beatmap generation also requires prediction of 2D
positioning of notes. Adding further complexity,

- The same exact audio can be mapped to wildly different types
  of notes and patterns depending on intended difficulty for the
  player and the map creator's creativity, which makes beatmap
  prediction more similar to music/art generation than automatic
  transcription

- Ideal beatmaps contain certain note placement motifs and patterns
  such as "streams" and regularly spaced "jumps" that improve map
  enjoyment for players, and an automatic beatmap generator should
  aim to be able to generate these patterns

_________________

Previously, github user kotritrona has done previous work on beatmap
prediction. A link to the repository is here:
https://github.com/kotritrona/osumapper

However, their model classifies one note at a time, which may not be
able to capture long-term patterns related to music sections
(intro, pre-chorus, chorus, etc) that can be seen in human-created
beatmaps.

Leveraging recent advances using the Transformer model in NLP,
I attempt to frame automatic beatmap generation as an NLP problem
and predict beatmap notes using an attention-based sequence-to-sequence
model, where the notes are embedded as "words".

The word embeddings contain note type and note
position information, while music audio is embedded as Mel-spectrograms.
Notes embeddings have further incorporated contextual information by
learning vector representations of the words using Word2Vec.

Similar work on translating between audio and text spaces has been
done before in the automatic speech recognition (ASR) domain, and 
tranformers were found to work well.




Thanks to Elessey for ranked beatmap dataset:
https://osu.ppy.sh/community/forums/topics/330552

Thanks to kotritrona for inspiration and previous work:
https://github.com/kotritrona/osumapper

