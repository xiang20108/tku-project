import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from tensorflow import keras
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout
from random import randint



df = pd.read_csv('output/training_data.csv', index_col = 'Date-Time')
df.index = pd.to_datetime(df.index)

def animation_(data):
    fig = plt.figure(figsize=(16, 9), dpi=100, tight_layout=True)
    ax = plt.axes(xlim=(0, len(data)), ylim=(min(data) - 0.01, max(data) + 0.01))
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    line, = ax.plot([], [], lw=2)
    
    def update(i):
        x = np.linspace(0, i, i+1)
        y = data[:i+1]
        line.set_data(x, y)
        return line,

    def init():
        line.set_data([], [])
        return line,
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), init_func=init, interval=80, blit=True, repeat=True)
    plt.show()
    ani.save(f'gif/{randint(0, 100)}.gif')
    
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

def inverse_label(y):
    y = np.argmax(y,axis=1)        
    y = encoder.inverse_transform(y)
    return y


callback = EarlyStopping(monitor='val_loss', patience=3, min_delta = 0.005)
n_steps = 20
n_features = len(df.columns)-2

def create_model():
    model = Sequential()

    model.add(LSTM(32, return_sequences=True, input_shape=(n_steps,n_features)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32,return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    adam = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model



date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
# print('Date :', date_list, end='\n\n')

label = 'label_sma'
model_path = 'models/lstm/'
for i in range(4, len(date_list)):
    X_train = df.loc[date_list[i-4]:date_list[i-1], 'L1-BidPrice':]
    X_test = df.loc[date_list[i], 'L1-BidPrice':]
    y_train = df.loc[date_list[i-4]:date_list[i-1], label]
    y_test = df.loc[date_list[i], label]
    
    
    ros = RandomOverSampler(random_state = 1)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    

    sc = StandardScaler()
    X_train_std = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_std = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
    


    print('\n\n')
    print('date :', date_list[i])
    print('training :', date_list[i-4 : i])
    
    X_split, y_split = TimeSeries_split(X_train_std, y_train, n_steps)
    y_split = transform_label(y_split)
    
    model = create_model()
    history = model.fit(
        X_split, y_split,
        validation_split=0.1,
        epochs=100, batch_size=32,
        callbacks=[callback],
   )
    model.save(f'{model_path}{date_list[i].replace("-", "")}.h5')

