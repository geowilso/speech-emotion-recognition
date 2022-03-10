from data import get_data #, clean_data
from extract_features import cut_or_pad, extract_mfcc

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from google.cloud import storage

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[UK][London][geowilso]speech_emotion_recognition_0"
STORAGE_LOCATION = '/models/'

class Trainer(object):

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y

        self.experiment_name = EXPERIMENT_NAME

    def set_model(self, lr):
        """defines the model as a class attribute"""

        self.model = Sequential()
        self.model.add(layers.Masking(mask_value = -1000., input_shape=(None,13)))
        self.model.add(layers.LSTM(128, return_sequences=True))
        self.model.add(layers.LSTM(64))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(6, activation='softmax'))

        optimiser = Adam(learning_rate=lr)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimiser,
                      metrics='acc')

        self.model.summary()

        return self.model

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def run(self, X_val, y_val):
        """set and train the pipeline"""

        self.set_model(0.001)

        es = EarlyStopping(patience=20,
                           monitor='val_loss',
                           restore_best_weights=True)

        self.model.fit(self.X,
                       self.y,
                       callbacks=es,
                       batch_size=32,
                       epochs=50)

    def evaluate(self, X_test, y_test):
        """evaluates the model on test data"""
        return self.model.evaluate(X_test, y_test)


    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.h5')

    def save_model(self):
        self.model.save('model.h5')
        self.upload_model_to_gcp()

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)


    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == '__main__':
    df = get_data()
    X, y, le = extract_mfcc(df, n_mfcc=13, n_fft=2048, hop_length=512, sr=44100)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    #trainer = Trainer(X=X_train, y=y_train)
    #trainer.set_experiment_name('speech_emotion_model_1')
    #trainer.run(X_test, y_test)
    #trainer.evaluate(X_test, y_test)
    #trainer.save_model()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.2)

    trainer = Trainer(X=X_train, y=y_train)
    trainer.run(X_test, y_test)
    trainer.evaluate(X_test, y_test)
