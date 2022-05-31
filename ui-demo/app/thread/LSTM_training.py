import threading
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,LSTM
from time import localtime, strftime
from random import shuffle
import os
import shutil


class LSTM_training(threading.Thread):
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
            self.training()
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

        for i in range(3):
            self.all_parameters[f'layer{i+1}'] = self.all_parameters[f'layer{i+1}'].replace('%28', '(').replace('%29', ')')
        self.all_parameters['outputLayer'] = self.all_parameters['outputLayer'].replace('%28', '(').replace('%2C', ',').replace('+', ' ').replace('%3D', '=').replace('%27', "'").replace('%29', ')')

        self.keys = self.all_parameters.keys()

        parameter_str = ''
        for key in self.keys:
            parameter_str += key + ' : ' + str(self.all_parameters[key]) + '<br>'
        self.send(json.dumps({
            'messageType': 'text',
            'content': f'<p><span style="color:orange">[Your Parameters]</span><br>{parameter_str}</p>'
        }))

    def create_model(self, n_steps, n_features):
        model = Sequential()

        model.add(LSTM(32, return_sequences=True, input_shape=(n_steps,n_features)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32,return_sequences=False))
        model.add(Dense(3, activation='softmax'))
        adam = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()
        return model

    def create_animation(self, data):
        fig = plt.figure(figsize=(16, 9), dpi=100, tight_layout=True)
        ax = plt.axes(xlim=(0, len(data)), ylim=(min(data) - 0.01, max(data) + 0.01))
        plt.grid()
        plt.title('Loss curve ')
        plt.xlabel('iter')
        plt.ylabel('loss value')
        line, = ax.plot([], [], lw=2)
        
        def update(i):
            x = np.linspace(0, i, i+1)
            y = data[:i+1]
            line.set_data(x, y)
            return line,

        def init():
            line.set_data([], [])
            return line,
    
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), init_func=init, interval=100, blit=True, repeat=True)
        filename = self.thread_id + strftime("%Y%m%d%H%M%S", localtime()) + '.gif'
        ani.save(f'app/temp/{filename}')
        return filename

    def create_png(self, loss, val_loss):
        plt.figure(figsize=(16, 9), dpi=100)
        plt.title('Loss curve ')
        plt.ticklabel_format(style='plain')
        plt.xlabel('iter')
        plt.ylabel('loss value')
        plt.grid()
        plt.plot(loss, label = 'loss')
        plt.plot(val_loss, label = 'validation loss')
        plt.legend(loc='upper left')
        filename = self.thread_id + strftime("%Y%m%d%H%M%S", localtime()) + '.png'
        plt.savefig(f'app/temp/{filename}', format = 'png')
        return filename

    def training(self):
        df = pd.read_csv('app/training_data/training_data.csv', index_col='Date-Time')
        df.index = pd.to_datetime(df.index)
        label = 'label_sma' if self.all_parameters['label'] == 'mean' else 'label'
        
        def TimeSeries_split(X_input,y_input,n):
            X = []
            y = []
            for i in range(0,len(X_input)-n):
                X.append(X_input.iloc[i:i+n].values)
                y.append(y_input[:][i+n-1])
                
            X = np.array(X)
            y = np.array(y)
            return X,y


        encoder = LabelEncoder()
        def transform_label(y):
            encoder.fit(y)
            encoded_y = encoder.transform(y)
            y = np_utils.to_categorical(encoded_y)
            
            return y


        callback = EarlyStopping(monitor='val_loss', patience=3)
        n_steps = 20
        n_features = len(df.columns)-2


        date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
        os.makedirs(f'app/temp/{self.create_time + self.thread_id}', mode=777)
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
            
            model = self.create_model(n_steps, n_features)
            history = model.fit(X_split, y_split,
                validation_split=0.1,
                epochs=100, batch_size=32,
                callbacks=[callback],
                verbose = '2'
            )
            
            model.save(f'app/temp/{self.create_time + self.thread_id}/{date_list[i].replace("-", "")}.h5')
            
            message = ''
            message += f'''
                model for : {date_list[i]}<br>
                training date : {date_list[i-4 : i]}<br><br>
            '''
            for j in range(len(history.history['loss'])):
                message += f'''
                    [Epoch {j+1: 2d}]&nbsp;&nbsp;loss : <span style = "color: #00ff00">{history.history['loss'][j]: .3f}</span>&nbsp;&nbsp;&nbsp;&nbsp;
                    accuracy : <span style = "color: #00ff00">{history.history['accuracy'][j]: .3f}</span>&nbsp;&nbsp;&nbsp;&nbsp;
                    val_loss : <span style = "color: #00ff00">{history.history['val_loss'][j]: .3f}</span>&nbsp;&nbsp;&nbsp;&nbsp;
                    val_accuracy : <span style = "color: #00ff00">{history.history['val_accuracy'][j]: .3f}</span><br>
                '''
            message += f'epoch : {len(history.history["loss"])}'
            self.send(json.dumps({
                'messageType': 'text',
                'content': f'<p>{message}</p>'
            }))
            

            # loss_figure = self.create_png(history.history['loss'], history.history['val_loss'])
            
            # self.send(json.dumps({
            #     'messageType': 'showLossCurve',
            #     'filename': loss_figure
            # }))
                

            if self.running == False:
                return
        self.save_model()

    def save_model(self):
        shutil.move(f'app/temp/{self.create_time + self.thread_id}', f'app/training_models/LSTM/')
        self.send(json.dumps({
                'messageType': 'text',
                'content': f'<p>Models save as <span style = "color:#00ff00">{self.create_time + self.thread_id}</span></p>'
        }))