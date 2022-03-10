# IMPORTS
import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import sounddevice as sd
import time
# import soundfile as sf

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import matplotlib.pyplot as plt

# to play the audio files
from IPython.display import Audio
import warnings
from tensorflow.keras.models import load_model

# model
from predict import record, processing, model_predict, playback

#PAGE CONFIG
st.set_page_config(
     page_title="Record Your Emotions",
     page_icon="🎥",
     layout="wide",
     initial_sidebar_state="expanded",
 )

#TITLE
st.markdown("<h1 style='text-align: center; color: darkred;'>Record Your Emotions</h1>", unsafe_allow_html=True )


#RECORD BUTTON
sound = np.empty
if st.button('Record', help='record your emotions'):
    with st.spinner('Recording for 3 seconds ....'):
        sound = record()
        time.sleep(5)
    if sound.any():
        st.success("Recording completed")


#PLAYBACK BUTTON
if st.button('Play Recording', help='playback your audio'):
    print(sound)
    playback(sound)



#UPLOAD BUTTON
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)



# load model
model = load_model("/models/speech_emotion_model_0.h5")


#TRYING TO PRINT A SOUND SIGNAL
# CREMA = '../raw_data/wav_files/'
# dir_list = os.listdir(CREMA)
# st.write(dir_list[0:10])
# file_name = CREMA + '1015_IEO_HAP_HI.wav'
# audio_file = open(file_name, 'rb')
# audio_bytes = audio_file.read()
# st.write("This is your audio")
# st.audio(audio_bytes, format='audio/ogg')


col1, col2, col3 = st.columns(3)
with col1:
    st.header("A MEL_SPECTOGRAM")
    st.markdown("![Alt Text](https://media.giphy.com/media/vybWlRniCXzZC/giphy.gif)")

with col2:
    st.header("A MAIN EMOTION")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("DETAILED EMOTIONS")
    st.image("https://static.streamlit.io/examples/owl.jpg")
