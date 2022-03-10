import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

def draw_mel(uploaded_file):
    sr = 44100
    file = librosa.load(uploaded_file, sr=sr)
    D = librosa.stft(file[0])  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Now with labeled axes!')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig
