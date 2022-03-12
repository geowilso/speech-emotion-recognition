FROM python:3.8.6-buster

COPY speech-emotion-recognition /speech-emotion-recognition
COPY models/speech_emotion_model_0.h5 models/speech_emotion_model_0.h5
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD streamlit run /speech-emotion-recognition/front_end.py
