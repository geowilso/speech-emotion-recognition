from data import get_data #, clean_data
import warnings
from google.cloud import storage

import numpy as np
import io
import soundfile as sf

from params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import librosa

warnings.filterwarnings("ignore")

def cut_or_pad(array, n):
    if array.shape[1] < n:
        return pad_sequences(array,n,padding='post',value=-1000.)
    elif array.shape[1] > n:
        return array[:,:n]
    else:
        return array


def extract_mfcc(df,
                 n_mfcc=13,
                 n_fft=2048,
                 hop_length=512,
                 sr = 44100):

    y = []
    X = []

    for i in tqdm(range(len(df))):
      y.append(df['emotion'][i])
      wav = librosa.load(df['path'][i], sr=sr)
      mfcc = librosa.feature.mfcc(wav[0], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
      mfcc = mfcc.T
      X.append(np.asarray(mfcc))

    X = np.asarray(X)
    y = np.asarray(y)

    return X,y

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
    print(df.head())
    print(df.shape)
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
    X.save
