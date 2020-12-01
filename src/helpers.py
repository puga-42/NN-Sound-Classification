import os
from PIL import Image
import pathlib
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import moviepy.editor as mp
import re
from pytube import YouTube
from moviepy.editor import *

# import keras
# from keras import layers
# from keras import layers
# from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')




def download_video(vid, folder):
    link = 'https://www.youtube.com/watch?v=' + vid
    folder = folder
    y = YouTube(link)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path=folder)
    return os.path.basename(os.path.normpath("/Users/joshbernd/Desktop/gal_notes/Capstone/Capstone_2/downloaded_videos/" + str(vid)))


def to_mp3(path, vid):
    for file in [n for n in os.listdir(path) if re.search('mp4',n)]:
        full_path = os.path.join(path, file)
#        output_path = os.path.join(download_folder, os.path.splitext(file)[0] + '.mp3')
        output_path = os.path.join(download_folder, str(vid) + '.mp3')
        
        clip = mp.AudioFileClip(full_path).subclip(10,) # disable if do not want any clipping
        clip.write_audiofile(output_path)
        
        return output_path, full_path

