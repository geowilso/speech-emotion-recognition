# IMPORTS
from re import A
import streamlit as st
import numpy as np

import pandas as pd
import os
import sys
import time
#import soundfile as sf
import sounddevice as sd

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import matplotlib.pyplot as plt
# trial
# to play the audio files
from IPython.display import Audio
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model
from predict import record, processing, model_predict, playback, grab_chunks
from graphs import draw_mel, plot_chunks
from helper import read_audio, record, save_record

#PAGE CONFIG
st.set_page_config(
     page_title="speech-emotion-recognition",
     page_icon="🎥",
     layout="wide",
     initial_sidebar_state="expanded",
 )

#TITLE
st.markdown(
    "<h1 style='text-align: center; color: #a6d1ff;'>Speech Emotion Recognition</h1>",
    unsafe_allow_html=True)

# st.text(len(sd.query_devices()))

st.markdown(
    "<h1 style='text-align: left; color: #a6d1ff;'>Either record your own voice here:</h1>",
    unsafe_allow_html=True)
#st.header("1. Record your own voice")

#RECORD BUTTON
if st.button(f"Click to Record"):
    record_state = st.text("Recording for 5 seconds...")
    duration = 5  # seconds
    fs = 44100
    myrecording = record(duration, fs)
    record_state.text(f"Saving sample as test.wav")

    uploaded_file = f"./temporary_recording/test.wav"

    save_record(uploaded_file, myrecording, fs)
    record_state.text(f"Done! Saved sample as test.wav")

    st.audio(read_audio(uploaded_file))

else:
    st.markdown(
        "<h1 style='text-align: left; color: #a6d1ff;'>Or upload an existing wav file here:</h1>",
        unsafe_allow_html=True)
    #UPLOAD BUTTON
    uploaded_file = st.file_uploader("")

if uploaded_file is not None:

    df, wav = grab_chunks(uploaded_file)
    fig = plot_chunks(df)

    col1, col2 = st.columns(2)
    with col1:
        fig

    with col2:
        a = df['Angry'].mean()
        h = df['Happy'].mean()
        n = df['Neutral'].mean()
        s = df['Sad'].mean()

        output_list = [a,h,n,s]
        emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
        colours = ['#ff9c9f', '#ffe7ab', '#d1d1d1', '#a6d1ff']
        mean_df = pd.DataFrame(output_list,columns=['result'])
        mean_df['emotion'] = emotions
        mean_df['colour'] = colours
        mean_df['percent'] = mean_df['result'].apply(
            lambda x: str(round(x * 100, 1)) + '%')
        mean_df = mean_df.sort_values(by='result', ascending=False)
        mean_df = mean_df.reset_index()

        size = 100

        for i in range(4):
            result_text = f"<h1 style='text-align: center; color: {mean_df['colour'][i]}; font-size: {size}px;'>{mean_df['emotion'][i]} {mean_df['percent'][i]}</h1>"
            st.markdown(result_text, unsafe_allow_html=True)
            size -= 25

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<h1 style='text-align: center; color: #a6d1ff;'>Melspectrogram</h1>",
            unsafe_allow_html=True)
        fig = draw_mel(wav[0])
        fig
    with col2:
        st.markdown(
            "<h1 style='text-align: center; color: #a6d1ff;'>Explanations</h1>",
            unsafe_allow_html=True)

    with col3:
        st.markdown(
            "<h1 style='text-align: center; color: #a6d1ff;'>Other stuff</h1>",
            unsafe_allow_html=True)



    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     plot1 = draw_mel(X[1][0])
    #     st.pyplot(plot1)

    # with col2:
    #     plot2 = plot_chunks(uploaded_file)
    #     st.pyplot(plot2)



# col1, col2, col3 = st.columns(3)
# with col1:
#     st.header("A MEL_SPECTOGRAM")
#     st.markdown("![Alt Text](https://media.giphy.com/media/vybWlRniCXzZC/giphy.gif)")
