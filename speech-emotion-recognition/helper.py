import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
#import wavio
from scipy.io.wavfile import write

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes


def record(duration, fs):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording


def save_record(path_myrecording, myrecording, fs):
    write(
        path_myrecording,
        fs,
        myrecording
    )
    return None
