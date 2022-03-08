from unittest import result
import sounddevice as sd
import soundfile as sf
import librosa
import numpy
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


def processing(myrecording):
    sound = myrecording.reshape(len(myrecording))
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    sr = 44100
    mfcc = librosa.feature.mfcc(sound,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)
    mfcc = mfcc.T
    mfcc.shape
    mfcc_T = mfcc.T
    mfcc_pad = pad_sequences(mfcc_T,
                            maxlen=615,
                            dtype='float32',
                            padding='post',
                            value=-1000.)
    mfcc_pad_T = mfcc_pad.T
    mfcc_pad_T_reshape = mfcc_pad_T.reshape(1, 615, 13)
    return mfcc_pad_T_reshape

def model_predict(mfcc_pad_T_reshape):
    model = load_model("../models/speech_emotion_model_0.h5")
    results= model.predict(mfcc_pad_T_reshape)
    return results
