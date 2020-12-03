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
import numpy as np
import pandas as pd
import moviepy.editor as mp
import re
from pytube import YouTube
from moviepy.editor import *
from sklearn.preprocessing import MultiLabelBinarizer
import glob
# import keras
# from keras import layers
# from keras import layers
# from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')



def data_to_df():
    dataset = pd.read_csv('data/balanced_train_segments.csv', delimiter=', ', header=2)
    dataset.columns = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']

    class_labels = pd.read_csv('data/class_labels_indices.csv')

    true_counts = pd.read_csv('data/qa_true_counts.csv')
    true_counts['sound'] = class_labels['display_name']
    true_counts['ratio'] = true_counts['num_true'] / true_counts['num_rated']

    return dataset, class_labels, true_counts



def explode_columns(x):
    x = x.strip('"')
    x = x.split(',')
    return x

def clean_dataset(df, mlb):
    df['positive_labels']
    df['recorded_labels'] = df['positive_labels'].map(explode_columns)
    
    
    mlb.fit_transform(df['recorded_labels'])
    return df


def clean_true_counts(df, class_labels):
    df = pd.read_csv('data/qa_true_counts.csv')
    df['sound'] = class_labels['display_name']

    df['ratio'] = df['num_true'] / df['num_rated']
    return df



def get_most_accurate(dataset, true_counts, ratio, mlb):
    true_counts = true_counts[true_counts['ratio'] < ratio]
    ##get array of all sounds we won't use
    sounds_not_used = np.array(true_counts['label_id'])
    
    
    ##binarize df
    df = dataset.join(pd.DataFrame(mlb.fit_transform(dataset.pop('recorded_labels')),
                                 columns=mlb.classes_,index=dataset.index))
    
    df = df.drop(columns=sounds_not_used)
    return df

def reduce_rows(df):
    numeric = df.drop(axis=1, columns=['start_seconds', 'end_seconds', 'positive_labels', 'YTID'])
    for idx in numeric.index:
        if sum(numeric.iloc[idx]) == 0:
            df = df.drop(index=idx, axis=0)

    return df





def download_video(vid, folder):
    link = 'https://www.youtube.com/watch?v=' + vid
    folder = folder
    y = YouTube(link)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path=folder)
    
    return os.path.basename(os.path.normpath("/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/mp4/" + str(vid)))
    


def to_mp3(path, vid):
    for file in [n for n in os.listdir(path)]:
        full_path = os.path.join(path, file)
        output_path = os.path.join("/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/mp3", 
                                    str(vid) + '.mp3')
        clip = mp.AudioFileClip(full_path).subclip(10,) # disable if do not want any clipping
        clip.write_audiofile(output_path)
        os.remove(full_path)
        return output_path, full_path


def cleanup():
    files = glob.glob('/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/mp4/*')
    for f in files:
        os.remove(f)

    files = glob.glob('/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/mp3/*')
    for f in files:
        os.remove(f)


def make_feature_file():
    header = 'YTID chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()
    file = open('dataset.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)


def get_features(path, start):
#     for filename in os.listdir(path):
#         vid = path + f'{filename}'
#         if songname != path + '/.DS_Store':
    mp3_file = os.path.basename(path)
    filename = os.path.splitext(mp3_file)[0]


    y, sr = librosa.load(path, mono=True)
    y = y[sr*start:(sr*start+10)]
    #y = y[sr*start:sr*(start+10)]

    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        
    return to_append
            
def write_to_file(row):            
    file = open('dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(row.split())