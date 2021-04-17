### Import required libraries
import numpy as np
import pandas as pd
import librosa
import os
import fnmatch
import json


n_mfcc = 30


### Function to shift the data along time axis
def manipulate(data, sr, time, direction):
    shift = int(sr*time)
    if direction == 'right':
        shift = -shift
    aug_data = np.roll(data,shift)
    if shift > 0:
        aug_data[:shift] = 0
    else:
        aug_data[shift:] = 0
    return aug_data


### Function to chop initial and end parts of the audio file
def crop(data, sr, time):
    data = manipulate(data, sr, time, 'right')
    data = manipulate(data, sr, time*2, 'left')
    data = manipulate(data, sr, time, 'right')
    return data


### Function to create 2 different data frames
### 1. Dataframe containing only MFCCs
### 2. Dataframe containing mean values of MFCCs along with other audio features
def create_df(folders, columns, types,audio_clip = 3, n_mfcc = 20):
    features = []
    mfccs = {
        'mfcc': [],
        'type': []
    }
    index = 0
    for folder in folders:
        for name in types:
            files = fnmatch.filter(os.listdir(folder), name)
            label = name.split("*")[0]
            for file in files:
                x, sr = librosa.load(folder+file, sr=22050)
                x = crop(x, sr, 0.3)
                time = int(librosa.get_duration(x, sr)) * sr
                clip = audio_clip
                for i in range(0, time+1 - sr*clip, sr*clip):
                    ### get MFCCs
                    mfcc = librosa.feature.mfcc(x[i:i+sr*clip], sr=sr, n_mfcc=n_mfcc)
                    mfccs['mfcc'].append(mfcc.T.tolist())
                    mfccs['type'].append(label)
                    features.append([np.mean(x) for x in mfcc])
                    features[index].append(sum(librosa.zero_crossings(x[i:i+sr*3])))
                    features[index].append(np.mean(librosa.feature.spectral_centroid(x[i:i+sr*3])))
                    features[index].append(np.mean(librosa.feature.spectral_rolloff(x[i:i+sr*3],sr=sr)))
                    features[index].append(np.mean(librosa.feature.chroma_stft(x[i:i+sr*3],sr=sr)))
                    features[index].append(label)
                    index += 1
    return pd.DataFrame(features, columns=columns), mfccs



### Similar function as above but doesnt split the audio clips
def create_df_without_clips(folders, columns, types, n_mfcc = 20):
    features = []
    mfccs = {
        'mfcc': [],
        'type': []
    }
    index = 0
    for folder in folders:
        for name in types:
            files = fnmatch.filter(os.listdir(folder), name)
            label = name.split("*")[0]
            for file in files:
                x, sr = librosa.load(folder+file, sr=22050)
                x = crop(x, sr, 0.3)
                mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc)
                mfccs['mfcc'].append(mfcc.T.tolist())
                mfccs['type'].append(label)
                features.append([np.mean(x) for x in mfcc])
                features[index].append(sum(librosa.zero_crossings(x)))
                features[index].append(np.mean(librosa.feature.spectral_centroid(x)))
                features[index].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
                features[index].append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
                features[index].append(label)
                index += 1
    return pd.DataFrame(features, columns=columns), mfccs



### generate the column names
columns = ['mfcc_'+str(i) for i in range(n_mfcc)]
for feature in ['zero', 'centroid', 'rolloff', 'chroma', 'type']:
    columns.append(feature)
print("Sucessfully created the columns")

### Extract all types of heartbeat sounds
types = ['normal*.wav', 'artifact*.wav', 'murmur*.wav', 'extrahls*.wav', 'extrastole*.wav']
print("Sucessfully identified all the types of heartbeat sounds")

### Folder names
folders = ['data/set_a/new/', 'data/set_b/new/']

### Create the required dataframes
print("Started reading and extracting audio features without clipping")
data_without_clips, mfcc_without_clips = create_df_without_clips(folders, columns, types, n_mfcc=30)
print("Sucessfully completing reading and extracting audio features without clipping")

print("Started reading and extracting audio features with clipping")
data, mfcc = create_df(folders, columns, types, n_mfcc=30, audio_clip=3)
print("Sucessfully completing reading and extracting audio features with clipping")

### Write data to files
data.to_csv("Features.csv")
print("Sucessfully written features to Features.csv file")
with open("MFCC.json", 'w') as fp:
    json.dump(mfcc, fp, indent = 4)
print("Sucessfully written mfccs into MFCC.json file")

data_without_clips.to_csv("Features_without_clips.csv")
print("Sucessfully written features to Features_without_clips.csv file")

with open("MFCC_without_clips.json", 'w') as fp:
    json.dump(mfcc, fp, indent = 4)
print("Sucessfully written mfccs into MFCC_without_clips.json file")

print("Sucessfully completed all steps!!!")