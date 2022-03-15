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


#new
import base64
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events



# #PAGE CONFIG
# st.set_page_config(
#     page_title="speech-emotion-recognition",
#     page_icon="🎥",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

#TITLE
st.markdown(
    "<h1 style='text-align: center; color: #a6d1ff;'>Speech Emotion Recognition</h1>",
    unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: left; color: #a6d1ff;'>Either record your own voice here:</h1>",
    unsafe_allow_html=True)
#st.header("1. Record your own voice")


stt_button = Button(label="Click to Record", width=100)

stt_button.js_on_event(
    "button_click",
    CustomJS(code="""
const timeMilliSec = 5000 //Fixed 5sec recording ... change here the value
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    const audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      //convert audioBuffer to wav
      const audioBlob = new Blob(audioChunks, {type:'audio/wav'});
      //create base64 reader
      var reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = function() {
        //read base64
        var base64data = reader.result;
        //send data to streamlit
        document.dispatchEvent(new CustomEvent("GET_AUDIO_BASE64", {detail: base64data}));
      }
    });
    setTimeout(() => {
      mediaRecorder.stop();
    }, timeMilliSec);
  });
  """))

result = streamlit_bokeh_events(stt_button,
                                events="GET_AUDIO_BASE64",
                                key="listen",
                                refresh_on_update=False,
                                override_height=75,
                                debounce_time=0)



if result:
    if "GET_AUDIO_BASE64" in result:
        b64_str_metadata = result.get("GET_AUDIO_BASE64")
        metadata_string = "data:audio/wav;base64,"
        if len(b64_str_metadata) > len(metadata_string):
            #get rid of metadata (data:audio/wav;base64,)

            if b64_str_metadata.startswith(metadata_string):
                b64_str = b64_str_metadata[len(metadata_string):]
            else:
                b64_str = b64_str_metadata

            decoded = base64.b64decode(b64_str)

            st.write("Read sound from Frontend")
            st.audio(decoded)

            #save it server side if needed
            with open('test.wav', 'wb') as f:
                f.write(decoded)

            file = open('test.wav', 'rb')

            uploaded_file = f"./temporary_recording/test.wav"
            save_record(uploaded_file, decoded, 44100)

            st.write("Read sound by saving in server and reloading file")
            st.audio(file)
#new



#RECORD BUTTON
# if st.button(f"Click to Record"):
#     record_state = st.text("Recording for 5 seconds...")
#     duration = 5  # seconds
#     fs = 44100
#     myrecording = record(duration, fs)
#     record_state.text(f"Saving sample as test.wav")

# uploaded_file = f"./temporary_recording/test.wav"

# save_record(uploaded_file, myrecording, fs)
# record_state.text(f"Done! Saved sample as test.wav")

# st.audio(read_audio(uploaded_file))

else:
    st.markdown(
        "<h1 style='text-align: left; color: #a6d1ff;'>Or upload an existing wav file here:</h1>",
        unsafe_allow_html=True)
    #UPLOAD BUTTON
    uploaded_file = st.file_uploader("")

if uploaded_file is not None:

    st.audio(uploaded_file)

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

        size = 50

        for i in range(4):
            result_text = f"<h1 style='text-align: center; color: {mean_df['colour'][i]}; font-size: {size}px;'>{mean_df['emotion'][i]} {mean_df['percent'][i]}</h1>"
            st.markdown(result_text, unsafe_allow_html=True)
            size -= 12
