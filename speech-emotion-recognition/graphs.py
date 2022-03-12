import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display

from predict import chunks, model_predict, load

def draw_mel(uploaded_file):

    # wav = load(uploaded_file)
    d = librosa.stft(uploaded_file)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

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
    array = np.array(chunk_preds)
    df = pd.DataFrame(array, columns=['Angry', 'Happy', 'Neutral', 'Sad'])

    fig = plt.figure(figsize=(15,8))

    plt.plot(
        'Angry',
        data=df,
        marker='o',
        markerfacecolor='blue',
        markersize=12,
        color='red',
        linewidth=4,
        label="Angry",
        linestyle='dashed',
    )

    plt.plot(
        'Happy',
        data=df,
        marker='o',
        markerfacecolor='blue',
        markersize=12,
        color='yellow',
        linewidth=4,
        label="Happy",
        linestyle='dashed',
    )

    plt.plot(
        'Neutral',
        data=df,
        marker='o',
        markerfacecolor='blue',
        markersize=12,
        color='grey',
        linewidth=4,
        label="Neutral",
        linestyle='dashed',
    )

    plt.plot(
        'Sad',
        data=df,
        marker='o',
        markerfacecolor='blue',
        markersize=12,
        color='blue',
        linewidth=4,
        label="Sad",
        linestyle='dashed',
    )

    plt.legend()
    plt.show()
