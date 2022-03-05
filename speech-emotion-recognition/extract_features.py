from Speech-Emotion-Recognition.data import get_data, clean_data

import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import librosa

def cut_or_pad(array, n):
    if array.shape[1] < n:
        return pad_sequences(array,n,padding='post',value=-1000.)
    elif array.shape[1] > n:
        return array[:,:n]
    else:
        return array


def extract_features(df,
                     sr=44100,
                     length=250,
                     offset=0.3,
                     n_mfcc=13,
                     poly_order=2,
                     n_chroma=10,
                     n_mels=10,
                     zcr=True,
                     rms=True):

    features = []

    for path in tqdm(df['path']):
        wav = librosa.load(path, sr=sr, offset=offset, duration = 10)

        mfcc = librosa.feature.mfcc(wav[0], sr=sr, n_mfcc=n_mfcc)
        array = cut_or_pad(mfcc,length)

        if poly_order != None:
            poly = librosa.feature.poly_features(y=wav[0],sr=sr,order=poly_order)
            poly = cut_or_pad(poly,length)

            array = np.vstack((array,poly))

        if n_chroma != None:
            chroma = librosa.feature.chroma_stft(y=wav[0],sr=sr,n_chroma=n_chroma)
            chroma = cut_or_pad(chroma,length)

            array = np.vstack((array,chroma))

        if n_mels != None:
            melspec = librosa.feature.melspectrogram(y=wav[0],sr=sr,n_mels=n_mels)
            melspec = cut_or_pad(melspec,length)
            log_S = librosa.amplitude_to_db(melspec)

            array = np.vstack((array,log_S))

        if zcr == True:
            zcr = librosa.feature.zero_crossing_rate(y=wav[0])
            zcr = cut_or_pad(zcr,length)

            array = np.vstack((array,zcr))

        if rms == True:
            rms = librosa.feature.rms(y=wav[0])
            rms = cut_or_pad(rms,length)

            array = np.vstack((array,rms))

        array = array.T

        features.append(array)

    X = pad_sequences(features, dtype='float32', padding='post', value=-1000.)
    return X

def targets(df):
    y = df['emotion']
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    y_cat = to_categorical(y_num)

    return y_cat

# def Xy_split(df,array):
#     y = df['emotion']
#     le = LabelEncoder()
#     y_num = le.fit_transform(y)
#     y_cat = to_categorical(y_num)

#     X_train, X_test, y_train, y_test = train_test_split(array, y_cat, test_size=0.15)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

#     return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == '__main__':
    df = get_data()
    X = extract_features(df,
                        sr=44100,
                        length=250,
                        offset=0.3,
                        n_mfcc=13,
                        poly_order=None,
                        n_chroma=None,
                        n_mels=None,
                        zcr=False,
                        rms=False)
    print(X.shape)
