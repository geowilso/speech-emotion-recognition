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

    #connect to bucket
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    #select crema folder from dataset
    crema = bucket.list_blobs(prefix='raw_data/wav_files/')
    crema_directory_list = []

    #crema load files from bucket as crema_directory_list
    for file in crema:
        file_name = file.name.split('wav_files/')
        if '.wav' in file_name[1]:
            crema_directory_list.append(file_name[1])

    crema_directory_list.sort()

    gender = []
    emotion = []
    path = []
    female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

    for i in crema_directory_list:
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
        path.append(
            'raw_data/wav_files/' + i)

    crema_df = pd.DataFrame(emotion, columns = ['emotion'])
    crema_df = pd.concat([crema_df,pd.DataFrame(gender, columns = ['gender'])],axis=1)
    crema_df = pd.concat([crema_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    crema_df['source'] = 'crema'

    #select tess folder from dataset
    tess = bucket.list_blobs(prefix='raw_data/tess/')
    tess_directory_list=[]

    for file in tess:
        file_name = file.name.split('tess/')
        if '.wav' in file_name[1]:
            tess_directory_list.append(file_name[1])
    path = []
    emotion = []
    tessname = []

    for i in tess_directory_list:
        tessname.append('raw_data/tess/' + i)

    for f in tessname:
        if 'OAF_angry' in f or 'YAF_angry' in f:
            emotion.append('angry')
        elif 'OAF_disgust' in f or 'YAF_disgust' in f:
            emotion.append('disgust')
        elif 'OAF_Fear' in f or 'YAF_fear' in f:
            emotion.append('fear')
        elif 'OAF_happy' in f or 'YAF_happy' in f:
            emotion.append('happy')
        elif 'OAF_neutral' in f or 'YAF_neutral' in f:
            emotion.append('neutral')
        elif 'OAF_Pleasant_surprise' in f or 'YAF_pleasant_surprised' in f:
            emotion.append('surprise')
        elif 'OAF_Sad' in f or 'YAF_sad' in f:
            emotion.append('sad')
        else:
            emotion.append('Unknown')
        path.append(f)

    tess_df = pd.DataFrame(emotion, columns = ['emotion'])
    tess_df['source'] = 'tess'
    tess_df = pd.concat([tess_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    tess_df['gender'] = 'female'


    #select ravdess folder from dataset
    ravdess = bucket.list_blobs(prefix='raw_data/ravdess/')
    ravdess_directory_list=[]

    for file in ravdess:
        file_name = file.name.split('ravdess/')
        if '.wav' in file_name[1]:
            ravdess_directory_list.append(file_name[1])

    emotion = []
    gender = []
    path = []
    fname = []
    for x in ravdess_directory_list:
        fname.append('raw_data/ravdess/' + x)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append('raw_data/ravdess/' + x)

    ravdess_df = pd.DataFrame(emotion)
    ravdess_df = ravdess_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    ravdess_df = pd.concat([pd.DataFrame(gender),ravdess_df],axis=1)
    ravdess_df.columns = ['gender','emotion']
    ravdess_df['source'] = 'ravdess'
    ravdess_df = pd.concat([ravdess_df,pd.DataFrame(path, columns = ['path'])],axis=1)


    savee = bucket.list_blobs(prefix='raw_data/savee/')
    savee_directory_list=[]

    for file in savee:
        file_name = file.name.split('savee/')
        if '.wav' in file_name[1]:
            savee_directory_list.append(file_name[1])

    emotion=[]
    path = []
    for i in savee_directory_list:
        if i[-8:-6]=='_a':
            emotion.append('angry')
        elif i[-8:-6]=='_d':
            emotion.append('disgust')
        elif i[-8:-6]=='_f':
            emotion.append('fear')
        elif i[-8:-6]=='_h':
            emotion.append('happy')
        elif i[-8:-6]=='_n':
            emotion.append('neutral')
        elif i[-8:-6]=='sa':
            emotion.append('sad')
        elif i[-8:-6]=='su':
            emotion.append('surprise')
        else:
            emotion.append('error')
        path.append('raw_data/savee/' + i)

    # Now check out the label count distribution
    savee_df = pd.DataFrame(emotion, columns = ['emotion'])
    savee_df['source'] = 'savee'
    savee_df = pd.concat([savee_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
    savee_df['gender'] = 'male'

    # emodb = bucket.list_blobs(prefix='raw_data/emodb/')

    targets = pd.concat([crema_df,tess_df,ravdess_df,savee_df])
    targets = targets[targets['emotion']!='surprise']
    targets = targets[targets['emotion']!='fear']
    targets = targets[targets['emotion']!='disgust']
    targets = targets.reset_index().drop(['index'], axis=1)
    return targets


if __name__ == '__main__':
    df = get_data()
    print(df.shape)
