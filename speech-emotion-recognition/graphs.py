import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from predict import chunks, model_predict

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

def plot_chunks(file):
    chunks = chunks(file)
    chunk_preds = []
    for chunk in chunks:
        chunk_pred = model_predict(chunk)
        chunk_preds.append(chunk_pred)
