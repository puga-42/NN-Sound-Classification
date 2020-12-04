# CNN-Instrument-Classification
Using CNNs to identify instruments in sound files




# The Data
I used google's Audioset metatdata. A file containing video ids of 22,176 youtube videos with a 10 second segment and up two 12 labels of sounds heard on that segment.


### Metatdata
The metadata consits of Youtube video ids 'YTID'

![alt text](img/metadata.png "Title")

### Class Labels
![alt text](img/class_labels.png "Title")


### Labeling Accuracy
Some of the labeled sounds don't actually appear in the video segments. I will cut the data down to the sounds with labeling accuray >= 80%. This corresponds to class lables with a ratio over 0.8.
![alt text](img/true_counts.png "Title")



Relevant features of the data:
    -


## Feature Extraction Pipeline

1. Download audio data from youtube videos in the metadata
2. Convert mp4 files to mp3
3. Trim mp3 file down to the 10 seconds indicated in the metadata.
4. Use the librosa library to generate a mel-spectrogram and extract the features
    - explain what mel-spectrogram is
    - Mel(f) = 2596 log(1 + f/700)
5. Append the list of features to the 'Features Dataframe'
    
    
## Let's visualize the feature extraction from a snippet of 'Linus and Lucy'
- ipd.Audio(file_deb)

![alt text](img/linus_and_lucy.png "Title")


## Features extracted from mel-spectrogram:
###     - Mel-frequency cepstral coefficients (MFCC)
  
###     - Spectral Centroid
    - A measure of where a spectrum's center of mass is located. This measurement is used to quantify the 'brightness' of a sound.
  ![alt text](img/spectral_centroid.png "Title")
    
###     - Zero Crossing Rate
    ![alt text](img/zcr.png "Title")

    
###     - Chroma Frequencies
    - Idendifies sounds that fall into distinced pitches. Large amounts of chroma features is a strong indicator for the presence of music.
   ![alt text](img/chroma_freq.png "Title")
     
    
###     - Spectral Roll-off
    - The frequency below which a certain percentage of the total spectral energy lies.
  ![alt text](img/spectral_rolloff.png "Title")
    

## Building the Neural Networks
Need to get a more even split sample of speech and no speech/music and no music

### Recognizing the Presence of Music

              
### Recognizing the Presence of Speech


## Neural Network Results
It is terrible. 

More data will need to be used to obtain more accurate results on a wider range of sounds.
