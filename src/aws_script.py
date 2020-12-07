from src.helpers import *

import pandas as pd
import numpy as np

df = pd.read_csv('data/df_reduced.csv')
path = "/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/mp4"

header = make_feature_file()


for video in df['YTID']:
    try: #Download video as mp4 
        filename = download_video(str(video), path)
    except:
        cleanup()
        continue

    mp3, mp4 = to_mp3(path, filename)
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

df.to_csv('/Users/joshbernd/Desktop/gal_notes/Capstone/CNN-Instrument-Classification/data/labels_df', index=False)
