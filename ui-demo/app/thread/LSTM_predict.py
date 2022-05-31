from keras.layers import Dense, LSTM
import shutil
import os
from random import shuffle
from time import localtime, strftime
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import threading
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler



class LSTM_predict(threading.Thread):
    def __init__(self, query_string: str, send):
        self.query_string = query_string
        self.send = send
        id_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        shuffle(id_list)
        self.thread_id = id_list.pop()
        self.create_time = strftime("%Y%m%d%H%M%S", localtime())
        self.running = True
        threading.Thread.__init__(self)

    def run(self):
        print('LSTM thread is running.')
        try:
            self.parameters_processing()
            self.predict()
            self.change_state()
        except Exception as e:
            self.send_error_message(e)        

    def stop(self):
        self.running = False

    def send_error_message(self, e):
        print(e)
        self.send(json.dumps({
            'messageType': 'text',
            'content': '<p><span style="color:red">[ERROR]</span> ' + str(e) + '</p>'
        }))
        self.change_state()

    def change_state(self):
        self.send(json.dumps({
            'messageType': 'done',
            'filename': 'done.png'
        }))

    def parameters_processing(self):
        query_string = self.query_string.split('&')
        query_string_spilt = [p.split('=') for p in query_string]
        self.all_parameters = {}
        for _ in query_string_spilt:
            self.all_parameters[_[0]] = _[1]

        self.keys = self.all_parameters.keys()

        parameter_str = ''
        for key in self.keys:
            parameter_str += key + ' : ' + \
                str(self.all_parameters[key]) + '<br>'
        self.send(json.dumps({
            'messageType': 'text',
            'content': f'<p><span style="color:orange">[Your Parameters]</span><br>{parameter_str}</p>'
        }))

    def predict(self):
        df = pd.read_csv('app/training_data/training_data.csv',
                         index_col='Date-Time')
        df.index = pd.to_datetime(df.index)
        self.df_output = df.loc[:, 'label':'L1-AskSize']
        self.df_output.insert(0, 'signal', np.nan)
        label = 'label_sma' if self.all_parameters['label'] == 'mean' else 'label'

        def TimeSeries_split(X_input, y_input, n):
            X = []
            y = []
            for i in range(0, len(X_input)-n):
                X.append(X_input.iloc[i:i+n].values)
                y.append(y_input[:][i+n-1])

            X = np.array(X)
            y = np.array(y)
            return X, y

        encoder = LabelEncoder()

        def transform_label(y):
            encoder.fit(y)
            encoded_y = encoder.transform(y)
            y = np_utils.to_categorical(encoded_y)

            return y

        def inverse_label(y):
            y = np.argmax(y, axis=1)
            y = encoder.inverse_transform(y)
            return y

        n_steps = 20

        date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
        for i in range(4, len(date_list)):
            X_train = df.loc[date_list[i-4]:date_list[i-1], 'L1-BidPrice':]
            X_test = df.loc[date_list[i], 'L1-BidPrice':]
            y_train = df.loc[date_list[i-4]:date_list[i-1], label]
            y_test = df.loc[date_list[i], label]

            # imbalance
            ros = RandomOverSampler(random_state=1)
            X_train, y_train = ros.fit_resample(X_train, y_train)

            # Standardalization
            sc = StandardScaler()
            X_train_std = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_std = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

            X_split, y_split = TimeSeries_split(X_train_std, y_train, n_steps)
            y_split = transform_label(y_split)

            model = keras.models.load_model(
                f'app/training_models/LSTM/{self.all_parameters["model"]}/{date_list[i].replace("-", "")}.h5')

            X_test_split, y_test_split = TimeSeries_split(X_test_std, y_test, n_steps)
            predictions = model.predict(X_test_split)
            y_pred = inverse_label(predictions).astype('float32')
            y_pred = np.insert(y_pred, 0, np.zeros(n_steps) + np.nan)
            predict_score = 1 - (y_test != y_pred).sum() / len(y_pred)
            self.df_output.loc[date_list[i], 'signal'] = y_pred

            message = f'''
                predict date : {date_list[i]}<br>
                training date : {date_list[i-4: i]}<br><br>
                Total examples : {len(y_test)}<br>
                Misclassified examples : {(y_test != y_pred).sum()}<br>
                Accuracy : <span style="color: #00ff00">{predict_score: .3f}</span>
            '''

            self.send(json.dumps({
                'messageType': 'text',
                'content': f'<p>{message}</p>'
            }))

            if self.running == False:
                return
        self.save_record()

    def save_record(self):
        now = localtime()
        filename = f'{strftime("%Y%m%d%H%M%S", now)}{self.thread_id}.csv'
        self.df_output.dropna(inplace=True)
        self.df_output.to_csv(f'app/predict_results/{filename}')

        self.send(json.dumps({
            'messageType': 'text',
            'content': f'<p>result sava as <span style = "color: #00ff00">{filename}<span>.</p>'
        }))
