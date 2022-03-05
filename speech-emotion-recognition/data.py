import pandas as pd
from google.cloud import storage

from params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import warnings


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix='raw_data/wav_files'))

    CREMA = BUCKET_TRAIN_DATA_PATH
    dir_list = os.listdir(CREMA)
    dir_list.sort()
    print(dir_list[0:10])
    gender = []
    emotion = []
    path = []
    female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
            1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

    for i in dir_list:
        part = i.split('_')
        if int(part[0]) in female:
            temp = 'female'
        else:
            temp = 'male'
        gender.append(temp)

        if part[2] == 'SAD':
            emotion.append('sad')
        elif part[2] == 'ANG':
            emotion.append('angry')
        elif part[2] == 'DIS':
            emotion.append('disgust')
        elif part[2] == 'FEA':
            emotion.append('fear')
        elif part[2] == 'HAP':
            emotion.append('happy')
        elif part[2] == 'NEU':
            emotion.append('neutral')
        else:
            emotion.append('unknown')
        path.append(CREMA + i)

    CREMA_df = pd.DataFrame(emotion, columns = ['emotion'])
    #CREMA_df['source'] = 'CREMA'
    CREMA_df = pd.concat([CREMA_df,pd.DataFrame(gender, columns = ['gender'])],axis=1)
    CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    return df


if __name__ == '__main__':
    df = get_data()
