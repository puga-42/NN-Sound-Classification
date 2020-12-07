import pandas as pd
import numpy as np
from helpers import *

df = pd.read_csv('data/df_reduced.csv')



path = "path to save mp4 file"

header = make_feature_file()

## Download video from df, convert to mp3, extract features

for video in df['YTID']:
    print(video)
    try: #Download video as mp4 
        filename = download_video(str(video), path)
    except:
        print(f'an error occurred downloading {video}')
        cleanup()
        continue

    print('path', path)
    mp3, mp4 = to_mp3(path, filename)
    print('mp3 path', mp3)
      #trim mp3, convert to png, get features
    start = int(df['start_seconds'][df['YTID'] == video])
    try:    
        features = get_features(mp3, start)
    except:
        continue
        
    try:
        write_to_file(features)
    except Exception:
        df = df[df['YTID'] != video]
        cleanup()
        continue
        
    cleanup()



train = pd.read_csv('features.csv')
labels = pd.read_csv('labels.csv')

##identify music

label_col = []
for vid in labels['YTID']:
    b = int(labels['/m/04rlf'][labels['YTID'] == vid])
    if b == 1:
        label_col.append(1)
    else:
        label_col.append(0)
y_music = np.array(label_col)


##Train test split
scaler = StandardScaler()
X = scaler.fit_transform(np.array(train.iloc[:, 1:-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y_music, test_size=0.2)

##make sequential model

model = Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
