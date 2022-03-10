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

def extract_mfcc(df,
                 n_mfcc=13,
                 n_fft=2048,
                 hop_length=512,
                 sr = 44100):

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    y = []
    X = []

    for i in tqdm(range(len(df))):
        y.append(df['emotion'][i])

        file = bucket.blob(df['path'][i])
        file_as_string = file.download_as_string()
        wav = sf.read(io.BytesIO(file_as_string))
        mfcc = librosa.feature.mfcc(wav[0], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        X.append(np.asarray(mfcc))

    X = np.asarray(X)
    X_pad = pad_sequences(X, dtype='float32', padding='post', value=-1000.)

    y = np.asarray(y)
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    y_cat = to_categorical(y_num)

    return X_pad, y_cat, le

if __name__ == '__main__':
    df = get_data()

    print(df.shape)

    X, y, le = extract_mfcc(df, n_mfcc=13, n_fft=2048, hop_length=512, sr=44100)
    print(X.shape)
    print(le.classes_)
