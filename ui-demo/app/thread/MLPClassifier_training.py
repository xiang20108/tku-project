import threading
import json
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import localtime, strftime
from random import shuffle
import pickle
import os
import shutil


class MLPClassifier_training(threading.Thread):
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
        print('MLPClassifier thread is running.')
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

        temp = self.all_parameters['hiddenLayerSizes'].replace('+', '').split('%2C')
        self.all_parameters['hiddenLayerSizes'] = [int(t) for t in temp]

        self.all_parameters['alpha'] = float(self.all_parameters['alpha'])

        self.all_parameters['batchSize'] = int(self.all_parameters['batchSize']) if self.all_parameters['batchSize'] != 'auto' else 'auto'

        self.all_parameters['learningRateInit'] = float(self.all_parameters['learningRateInit'])

        self.all_parameters['powerT'] = float(self.all_parameters['powerT'])

        self.all_parameters['maxIter'] = int(self.all_parameters['maxIter'])

        self.all_parameters['shuffle'] = True if self.all_parameters['shuffle'] == 'True' else False
        
        self.all_parameters['randomState'] = int(self.all_parameters['randomState']) if self.all_parameters['randomState'] != 'None' else None

        self.all_parameters['tol'] = float(self.all_parameters['tol'])

        self.all_parameters['warmStart'] = True if self.all_parameters['warmStart'] == 'True' else False

        self.all_parameters['momentum'] = float(self.all_parameters['momentum'])
        
        self.all_parameters['nesterovsMomentum'] = True if self.all_parameters['nesterovsMomentum'] == 'True' else False

        self.all_parameters['earlyStopping'] = True if self.all_parameters['earlyStopping'] == 'True' else False

        self.all_parameters['validationFraction'] = float(self.all_parameters['validationFraction'])

        self.all_parameters['beta1'] = float(self.all_parameters['beta1'])

        self.all_parameters['beta2'] = float(self.all_parameters['beta2'])

        self.all_parameters['epsilon'] = float(self.all_parameters['epsilon'])

        self.all_parameters['nIterNoChange'] = int(self.all_parameters['nIterNoChange'])

        self.all_parameters['maxFun'] = int(self.all_parameters['maxFun'])
        
        general_keys = ['label', 'hiddenLayerSizes', 'activation', 'solver', 'alpha', 'batchSize', 'maxIter', 'randomState', 'tol', 'warmStart', 'maxFun']

        if self.all_parameters['solver'] == 'adam':
            self.keys = general_keys + ['learningRateInit', 'shuffle', 'earlyStopping', 'validationFraction', 'beta1', 'beta2', 'epsilon', 'nIterNoChange']
        elif self.all_parameters['solver'] == 'sgd':
            self.keys = general_keys + ['learningRate', 'learningRateInit', 'powerT', 'shuffle', 'momentum', 'nesterovsMomentum', 'earlyStopping', 'validationFraction', 'nIterNoChange']
        elif self.all_parameters['solver'] == 'lbfgs':
            self.keys = general_keys

        parameter_str = ''
        for key in self.keys:
            parameter_str += key + ' : ' + str(self.all_parameters[key]) + '<br>'
        self.send(json.dumps({
            'messageType': 'text',
            'content': f'<p><span style="color:orange">[Your Parameters]</span><br>{parameter_str}</p>'
        }))

    def setting_model(self):
        mlp = MLPClassifier(
            hidden_layer_sizes = self.all_parameters['hiddenLayerSizes'],
            activation = self.all_parameters['activation'],
            solver = self.all_parameters['solver'],
            alpha = self.all_parameters['alpha'],
            batch_size = self.all_parameters['batchSize'],
            max_iter = self.all_parameters['maxIter'],
            random_state = self.all_parameters['randomState'],
            tol = self.all_parameters['tol'],
            warm_start = self.all_parameters['warmStart'],
            max_fun = self.all_parameters['maxFun']
        )
        if self.all_parameters['solver'] == 'adam':
            mlp.learning_rate_init = self.all_parameters['learningRateInit']
            mlp.shuffle = self.all_parameters['shuffle']
            mlp.early_stopping = self.all_parameters['earlyStopping']
            mlp.validation_fraction = self.all_parameters['validationFraction']
            mlp.beta_1 = self.all_parameters['beta1']
            mlp.beta_2 = self.all_parameters['beta2']
            mlp.epsilon = self.all_parameters['epsilon']
            mlp.n_iter_no_change = self.all_parameters['nIterNoChange']
        elif self.all_parameters['solver'] == 'sgd':
            mlp.learning_rate = self.all_parameters['learningRate']
            mlp.learning_rate_init = self.all_parameters['learningRateInit']
            mlp.power_t = self.all_parameters['powerT']
            mlp.shuffle = self.all_parameters['shuffle']
            mlp.momentum = self.all_parameters['momentum']
            mlp.nesterovs_momentum = self.all_parameters['nesterovsMomentum']
            mlp.early_stopping = self.all_parameters['earlyStopping']
            mlp.validation_fraction = self.all_parameters['validationFraction']
            mlp.n_iter_no_change = self.all_parameters['nIterNoChange']

        return mlp

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

    def training(self):
        df = pd.read_csv('app/training_data/training_data.csv', index_col='Date-Time')
        df.index = pd.to_datetime(df.index)
        label = 'label_sma' if self.all_parameters['label'] == 'mean' else 'label'

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
            

            mlp = self.setting_model()
            mlp.fit(X_train_std.values,y_train.values)
            
            
            with open(f'app/temp/{self.create_time + self.thread_id}/{date_list[i].replace("-", "")}', mode='wb') as f:
                pickle.dump(mlp, f)
            
            message = f'''model for : {date_list[i]}<br>training date : {date_list[i-4:i]}<br><br>
                    Epoch : { mlp.n_iter_}<br>
            '''
            if mlp.validation_fraction:  
                message += f'Best validation score : {mlp.best_validation_score_: .3f}<br>'
            if mlp.solver != 'lbfgs':
                message += f'Last loss : {mlp.loss_curve_[-1]: .4f}<br>'
            message += f'''Total examples : Total examples: {len(y_train)}<br>
                Misclassified examples: {(y_train.values != mlp.predict(X_train_std.values)).sum()}<br>
                Accuracy : <span style="color:#00ff00">{mlp.score(X_train_std, y_train): .3f}</span>'''

            if mlp.solver != 'lbfgs':
                loss_figure = self.create_animation(mlp.loss_curve_)


            self.send(json.dumps({
                'messageType': 'text',
                'content': f'<p>{message}</p>'
            }))

            if self.all_parameters['solver'] != 'lbfgs':
                self.send(json.dumps({
                'messageType': 'showLossCurve',
                'filename': loss_figure
            }))

            if self.running == False:
                return
        self.save_model()

    def save_model(self):
        shutil.move(f'app/temp/{self.create_time + self.thread_id}', f'app/training_models/MLPClassifier/')
        self.send(json.dumps({
                'messageType': 'text',
                'content': f'<p>Models save as <span style = "color:#00ff00">{self.create_time + self.thread_id}</span></p>'
            }))

