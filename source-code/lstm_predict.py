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
from keras.layers import Dense,LSTM
from time import strftime, localtime
import shutil

df = pd.read_csv('output/training_data.csv', index_col = 'Date-Time')
df.index = pd.to_datetime(df.index)
df_output = df.loc[:, 'label':'L1-AskSize']
df_output.insert(0, 'signal', np.nan)

    
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


n_steps = 20
n_features = len(df.columns)-2



date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
# print('Date :', date_list, end='\n\n')

model_path = 'models/lstm/'
label = 'label_sma'     
accuracy = []
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
    
    X_split, y_split = TimeSeries_split(X_train_std, y_train, n_steps)
    y_split = transform_label(y_split)

    model = keras.models.load_model(f'{model_path}{date_list[i].replace("-", "")}.h5')
    

    X_test_split, y_test_split = TimeSeries_split(X_test_std, y_test, n_steps)
    predictions = model.predict(X_test_split)
    y_pred = inverse_label(predictions).astype('float32')
    print('date :', date_list[i])
    print('training :', date_list[i-4 : i],end = '\n')
    print('\n<Test data>')
    print(f'Totol examples : {len(y_test)}')
    y_pred = np.insert(y_pred,0, np.zeros(n_steps) + np.nan)
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    predict_score = 1 - (y_test != y_pred).sum() / len(y_pred)
    print('Accuracy: %.3f' % predict_score)
    print('\n')
    accuracy.append(float(f'{predict_score: .3f}'))
    df_output.loc[date_list[i], 'signal'] = y_pred


df_output.dropna(inplace = True)   
df_output.to_csv('output/predict_result.csv')
dest_filename = strftime("%Y%m%d%H%M%S", localtime()) + 'K'
shutil.copyfile('output/predict_result.csv', f'../ui-demo/app/predict_results/{dest_filename}.csv')
print(f'\npredict result save as {dest_filename}')