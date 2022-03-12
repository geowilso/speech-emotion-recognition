from unittest import result
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def record():
    fs = 44100
    duration = 3  # seconds
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    print(type(myrecording))
    return myrecording

def playback(myrecording):
    sd.play(myrecording, 44100)
    sd.stop()


def load(uploaded_file):
    sr = 44100

    wav = librosa.load(uploaded_file, sr=sr)
    return wav


def processing(uploaded_file):

    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    sr = 44100

    wav = librosa.load(uploaded_file, sr=sr)

    mfcc = librosa.feature.mfcc(wav[0],
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)

    mfcc_pad = pad_sequences(mfcc,
                             maxlen=615,
                             dtype='float32',
                             padding='post',
                             value=-1000.)

    mfcc_pad_T = mfcc_pad.T
    mfcc_pad_T_reshape = mfcc_pad_T.reshape(1, 615, 13)
    return mfcc_pad_T_reshape


def chunks(uploaded_file):
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    sr = 44100

    wav = librosa.load(uploaded_file, sr=sr)

    mfcc = librosa.feature.mfcc(wav[0],
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)

    mfcc_list = []


    while len(mfcc_T) > 150:
        mfcc_chunk = mfcc_T[:150, :]
        mfcc_list.append(mfcc_chunk)
        mfcc_T = mfcc_T[150:, :]
    mfcc_list.append(mfcc_T)

    mfcc_list_pad = []

    for i in mfcc_list:
        mfcc = i.T
        mfcc_pad = pad_sequences(mfcc,
                                maxlen=615,
                                dtype='float32',
                                padding='post',
                                value=-1000.)
        mfcc_pad_T = mfcc_pad.T
        mfcc_pad_T_reshape = mfcc_pad_T.reshape(1, 615, 13)
        mfcc_list_pad.append(mfcc_pad_T_reshape)

    return mfcc_list_pad

def model_predict(mfcc_pad_T_reshape):
    model = load_model("models/speech_emotion_model_0.h5")
    results = model.predict(mfcc_pad_T_reshape)
    index = results.argmax(axis=1)
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    colours = ['#a81919', '#ffcb3b', '#9c9c9c', '#407eb8']
    df = pd.DataFrame(results.T,columns=['result'])
    df['emotion'] = emotions
    df['colour'] = colours
    df['percent'] = df['result'].apply(lambda x: str(round(x*100,1))+'%')
    df = df.sort_values(by='result', ascending=False)
    df = df.reset_index()
    return df
