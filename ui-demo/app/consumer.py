from channels.generic.websocket import WebsocketConsumer
from app.thread.MLPClassifier_training import MLPClassifier_training
from app.thread.MLPClassifier_predict import MLPClassifier_predict
from app.thread.LSTM_training import LSTM_training
from app.thread.LSTM_predict import LSTM_predict
from app.thread.Backtest_thread import Backtest_thread


class training_consumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        model_name = self.scope['url_route']['kwargs']['model_name']
        query_string = self.scope['query_string'].decode('utf-8')

        self.worker = None
        if model_name == 'MLPClassifier':
            self.worker = MLPClassifier_training(query_string, self.send)
        elif model_name == 'LSTM':
            self.worker = LSTM_training(query_string, self.send)
    
        if self.worker != None:
            self.worker.start()
        

        
    def disconnect(self, code):
        self.worker.stop()
        # self.worker.join()
        print('worker is stop')
        return super().disconnect(code)

class predict_consumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        model_name = self.scope['url_route']['kwargs']['model_name']
        query_string = self.scope['query_string'].decode('utf-8')

        self.worker = None
        if model_name == 'MLPClassifier':
            self.worker = MLPClassifier_predict(query_string, self.send)
        elif model_name == 'LSTM':
            self.worker = LSTM_predict(query_string, self.send)
        
        if self.worker != None:
            self.worker.start()
        

        
    def disconnect(self, code):
        self.worker.stop()
        # self.worker.join()
        print('worker is stop')
        return super().disconnect(code)


class backtest_consumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        query_string = self.scope['query_string'].decode('utf-8')
        self.worker = Backtest_thread(query_string, self.send)
        self.worker.start()

    def disconnect(self, code):
        self.worker.stop()
        # self.worker.join()
        print('worker is stop')
        return super().disconnect(code)

        