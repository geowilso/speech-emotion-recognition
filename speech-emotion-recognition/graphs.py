import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display


from predict import grab_chunks, model_predict, load


def draw_mel(wave):

    # wav = load(uploaded_file)
    d = librosa.stft(wave)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax, fmax=44100)
    #ax.set(title='Melspectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig


def plot_chunks(df):
    fig = plt.figure()

    plt.plot(
        'Angry',
        data=df,
        marker='o',
        markerfacecolor='#ff9c9f',
        markersize=12,
        color='#ff9c9f',
        linewidth=4,
        label="Angry",
        linestyle='dashed',
    )

    plt.plot(
        'Happy',
        data=df,
        marker='o',
        markerfacecolor='#ffe7ab',
        markersize=12,
        color='#ffe7ab',
        linewidth=4,
        label="Happy",
        linestyle='dashed',
    )

    plt.plot(
        'Neutral',
        data=df,
        marker='o',
        markerfacecolor='#d1d1d1',
        markersize=12,
        color='#d1d1d1',
        linewidth=4,
        label="Neutral",
        linestyle='dashed',
    )

    plt.plot(
        'Sad',
        data=df,
        marker='o',
        markerfacecolor='#a6d1ff',
        markersize=12,
        color='#a6d1ff',
        linewidth=4,
        label="Sad",
        linestyle='dashed',
    )

    # plt.legend()
    plt.show()

    return fig
