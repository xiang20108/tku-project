from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
import json
import os
import time

# Create your views here.
def home(request):
    title = 'Home : Service'
    cards = [
        {
            'id': 1,
            'name': 'Train',
            'image': 'images/training.png',
            'href': 'train/'
        },
        {
            'id': 2,
            'name': 'Predict',
            'image': 'images/predict.png',
            'href': 'predict/'
        },
        {
            'id': 3,
            'name': 'Backtest',
            'image': 'images/backtest.png',
            'href': 'backtest/'
        },
        {
            'id': 4,
            'name': 'Data',
            'image': 'images/database.png',
            'href': 'data/'
        }
    ]
    return render(request, 'index.html', locals())


def train(request):
    title = 'Models : Choose one'
    cards = [
        {
            'id': 1,
            'name': 'MLPClassifier',
            'image': 'images/scikit-learn.png',
            'href': 'MLPClassifier/'
        },
        {
            'id': 2,
            'name': 'LSTM',
            'image': 'images/keras.png',
            'href': 'LSTM/'
        },
    ]
    return render(request, 'index.html', locals())

def setting(request, model_name):
    title = f'{model_name} : Setting Parameters'
    with open('app/parameter/default_setting.json', mode = 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    data = data['general'] + data[model_name]
    return render(request, 'setting.html', locals())

def output(request, model_name):
    title = f'{model_name} : Training'
    return render(request, 'output.html', locals())

def predict(request):
    title = 'Models : Choose one'
    cards = [
        {
            'id': 1,
            'name': 'MLPClassifier',
            'image': 'images/scikit-learn.png',
            'href': 'MLPClassifier/'
        },
        {
            'id': 2,
            'name': 'LSTM',
            'image': 'images/keras.png',
            'href': 'LSTM/'
        },
    ]
    return render(request, 'index.html', locals())

def predict_setting(request, model_name):
    title = 'Predict : Setting Parameters'
    with open('app/parameter/default_setting.json', mode = 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    data = data['general']

    path = f'app/training_models/{model_name}/'
    files = os.listdir(path)
    files.remove('default')
    others = []
    if files:
        for f in files:
            others.append(f)
    model = {
        'type': 'option',
        'name': 'model',
        'value': {
                'default': 'default',
                'others': others
        },
        'label': 'Model'
    }
    data.append(model)
    return render(request, 'setting_center.html', locals())


def predict_output(request, model_name):
    title = f'{model_name} : Predict'
    return render(request, 'output.html', locals())

def backtest(request):
    title = 'Backtest : Setting Parameters'
    init_cash = {
        'type': 'text',
        'name': 'init_cash',
        'value': '10000000',
        'placeholder': 'int, ex.10000000',
        'label': 'Initial cash'
    }
    path = f'app/predict_results'
    others = []
    files = os.listdir(path)
    if files:
        default = files[-1]
        files.pop(-1)
        for f in files:
            others.append(f)
        records = {
            'type': 'option',
            'name': 'record',
            'value': {
                    'default': default,
                    'others': others
            },
            'label': 'Record'
        }
    else:
        records = {}

    return render(request, 'backtest.html', locals())

def backtest_result(request):
    title = f'Backtest : Result'
    return render(request, 'output.html', locals())


def data(request):
    title = 'Data : Choose one'
    cards = [
        {
            'id': 1,
            'name': 'JSON',
            'image': 'images/json.png',
            'href': 'json/'
        },
    ]
    return render(request, 'index.html', locals())

def return_data(request, format_):
    if format_ == 'json':
        with open('app/training_data/training_data.json', mode = 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        response = HttpResponse(json.dumps(data), content_type = 'application/json')
    return response

def resource(request, filename):
    if os.path.isfile(f'static/images/{filename}'):
        path = f'static/images/{filename}'
    elif os.path.isfile(f'app/temp/{filename}'):
        path = f'app/temp/{filename}'
    else:
        return HttpResponseNotFound('<h3 style="text-align:center"><span style="color:red">file does not exist.</span></h3>')

    with open(path, mode='rb') as f:
        img = f.read()
    file_type = filename.split('.')[1]
    return HttpResponse(img, content_type=f"image/{file_type}")