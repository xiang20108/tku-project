# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import localtime, strftime
import pickle



df = pd.read_csv('output/training_data.csv', index_col='Date-Time')
df.index = pd.to_datetime(df.index)


date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
# print('Date :', date_list, end='\n\n')

def animation_(data):
    fig = plt.figure(figsize=(16, 9), dpi=100,  tight_layout=True)
    ax = plt.axes(xlim=(0, len(data)), ylim=(min(data) - 0.01, max(data) + 0.01))
    plt.grid()
    plt.title('Loss curve ')
    plt.xlabel('epochs')
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
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), init_func=init, interval=120, blit=True, repeat=True)
    ani.save(f'gif/{strftime("%Y%m%d%H%M%S", localtime())}.gif')

label = 'label_sma' 
path = 'models/mlpclassifier/'
accuracy = []
for i in range(4, len(date_list)):
    X_train = df.loc[date_list[i-4]:date_list[i-1], 'L1-BidPrice':]
    X_test = df.loc[date_list[i], 'L1-BidPrice':]
    y_train = df.loc[date_list[i-4]:date_list[i-1], label]
    y_test = df.loc[date_list[i], label]
    

    ros = RandomOverSampler(random_state = 1)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print((y_train == 1).sum())
    print((y_train == 0).sum())
    print((y_train == -1).sum())

    sc = StandardScaler()
    X_train_std = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_std = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
     
    mlp = MLPClassifier(
        solver='adam', 
        hidden_layer_sizes=(32, 64, 32),
        learning_rate_init=0.001,
        early_stopping=True,
        shuffle= True,
        n_iter_no_change = 10,
        validation_fraction = 0.1,
        tol = 0.0001,
        max_iter = 500,
        verbose=False,
    )
    
    mlp.fit(X_train_std.values, y_train.values)
    with open(f'{path}{date_list[i].replace("-", "")}', mode = 'wb') as f:
        pickle.dump(mlp, f)

    
    print('\nmodel for :', date_list[i])
    print('training :', date_list[i-4 : i])
    print('epoch :', mlp.n_iter_)
    if mlp.validation_fraction and mlp.early_stopping:  
        print(f'Best validation score : {mlp.best_validation_score_: .3f}')
    if mlp.solver != 'lbfgs':
        print(f'last loss : {mlp.loss_curve_[-1]: .4f}')
    print(f'Total examples: {len(y_train)}')
    print('Misclassified examples: %d' % (y_train.values != mlp.predict(X_train_std.values)).sum())
    print('Accuracy: %.3f\n' % mlp.score(X_train_std.values, y_train.values))

