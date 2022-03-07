from Speech-Emotion-Recognition.data import get_data, clean_data
from Speech-Emotion-Recognition.extract_features import cut_or_pad, extract_features, targets

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_model(self, lr):
        """defines the model as a class attribute"""

        self.model = Sequential()
        self.model.add(layers.Masking(mask_value = -1000., input_shape=(self.X.shape[1], self.X.shape[2])))
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

    def run(self):
        """set and train the pipeline"""

        self.set_model(0.001)

        es = EarlyStopping(patience=20,
                           monitor='val_loss',
                           restore_best_weights=True)

        self.model.fit(self.X,
                       self.y,
                       callbacks=es,
                       batch_size=32,
                       epochs=100)

    def save(self):
        self.set_model(0.001)
        self.run()
        self.model.save('model')

    def evaluate(self, X_test, y_test):
        """evaluates the model on test data"""
        self.model.evaluate(X_test,y_test)



if __name__ == "__main__":
    df = get_data()
    #df = clean_data(df)

    X = extract_features(df,
                        sr=44100,
                        length=250,
                        offset=0.3,
                        n_mfcc=13,
                        poly_order=None,
                        n_chroma=None,
                        n_mels=None,
                        zcr=False,
                        rms=False)

    y = targets(df)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    trainer = Trainer(X_train,y_train)
    trainer.run()
    trainer.evaluate()
