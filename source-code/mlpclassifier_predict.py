# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

import pickle
from time import strftime, localtime
import shutil

df = pd.read_csv('output/training_data.csv', index_col='Date-Time')
df.index = pd.to_datetime(df.index)
df_output = df.loc[:, :'L1-AskSize']
df_output.insert(0, 'signal', np.nan)

date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
print('Date :', date_list, end='\n\n')

             
model_path = 'models/mlpclassifier/'
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
    
    with open(f'{model_path}{date_list[i].replace("-", "")}', mode = 'rb') as f:
        mlp = pickle.load(f)
    
    
    
    # predict
    y_pred = mlp.predict(X_test.values)
    df_output.loc[date_list[i], 'signal'] = y_pred
    print('date :', date_list[i])
    print('training :', date_list[i-4 : i],end = '\n')
    print('\n<Test data>')
    print(f'Total examples : {len(y_test)}')
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    predict_score = mlp.score(X_test.values,y_test.values)
    print('Accuracy: %.3f' % predict_score)
    print('\n')
    accuracy.append(float(f'{predict_score: .3f}'))
    

    


print('all accuracy :', accuracy)    
df_output.dropna(inplace = True)
df_output.to_csv('output/predict_result.csv')
dest_filename = strftime("%Y%m%d%H%M%S", localtime()) + 'K'
shutil.copyfile('output/predict_result.csv', f'../ui-demo/app/predict_results/{dest_filename}.csv')
print(f'\npredict result save as {dest_filename}')