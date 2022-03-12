# IMPORTS
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
from predict import record, processing, model_predict, playback
from graphs import draw_mel
from helper import read_audio, record, save_record

#PAGE CONFIG
st.set_page_config(
     page_title="Upload Your Emotions",
     page_icon="🎥",
     layout="wide",
     initial_sidebar_state="expanded",
 )

#TITLE
st.markdown("<h1 style='text-align: center; color: lightblue;'>Upload Your Emotions</h1>", unsafe_allow_html=True )

st.text(len(sd.query_devices()))

# #RECORD BUTTON
# sound = np.empty
# if st.button('Record', help='record your emotions'):
#     with st.spinner('Recording for 3 seconds ....'):
#         sound = record()
#         time.sleep(5)
#     if sound.any():
#         st.success("Recording completed")


# #PLAYBACK BUTTON
# if st.button('Play Recording', help='playback your audio'):
#     print(sound)
#     playback(sound)

st.header("1. Record your own voice")


if st.button(f"Click to Record"):
    record_state = st.text("Recording...")
    duration = 2  # seconds
    fs = 48000
    myrecording = record(duration, fs)
    record_state.text(f"Saving sample as test.wav")

    uploaded_file = f"./temporary_recording/test.wav"

    save_record(uploaded_file, myrecording, fs)
    record_state.text(f"Done! Saved sample as test.wav")

    st.audio(read_audio(uploaded_file))


    #UPLOAD BUTTON

else:
    st.header("2. Upload an existing file")
    uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:

    X = processing(uploaded_file)

    # st.text(wav[0].shape)
    # st.text(mfcc.shape)
    # st.text(mfcc_pad.shape)
    # st.text(mfcc_pad_T.shape)

    pred = model_predict(X)

    size = pred['result'][0] * 100
    result_text = f"<p style='font-family:sans-serif; color:{pred['colour'][0]}; font-size: {size}px;'>{pred['emotion'][0]} {pred['percent'][0]}</p>"
    st.markdown(result_text, unsafe_allow_html=True)

    size = pred['result'][1] * 100
    if size < 30:
        size = 30
    result_text = f"<p style='font-family:sans-serif; color:{pred['colour'][1]}; font-size: {size}px;'>{pred['emotion'][1]} {pred['percent'][1]}</p>"
    st.markdown(result_text, unsafe_allow_html=True)

    size = pred['result'][2] * 100
    if size < 20:
        size = 20
    result_text = f"<p style='font-family:sans-serif; color:{pred['colour'][2]}; font-size: {size}px;'>{pred['emotion'][2]} {pred['percent'][2]}</p>"
    st.markdown(result_text, unsafe_allow_html=True)

    size = pred['result'][3] * 100
    if size < 10:
        size = 15
    result_text = f"<p style='font-family:sans-serif; color:{pred['colour'][3]}; font-size: {size}px;'>{pred['emotion'][3]} {pred['percent'][3]}</p>"
    st.markdown(result_text, unsafe_allow_html=True)

    plot = draw_mel(uploaded_file)
    plot


# col1, col2, col3 = st.columns(3)
# with col1:
#     st.header("A MEL_SPECTOGRAM")
#     st.markdown("![Alt Text](https://media.giphy.com/media/vybWlRniCXzZC/giphy.gif)")
