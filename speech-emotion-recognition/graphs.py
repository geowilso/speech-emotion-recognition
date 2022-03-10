import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

def draw_mel(uploaded_file):
    file, sr = librosa.load(uploaded_file)
    D = librosa.stft(file)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Now with labeled axes!')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
