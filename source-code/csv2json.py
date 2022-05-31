import pandas as pd
import json


df = pd.read_csv('output/training_data.csv')
columns = df.columns

data = []

def convert(args):
    dict_ = {}
    for column in columns:
        dict_[column] = args[column]
    data.append(dict_)
    return

df = df.iloc[:50, :]
df.apply(convert, axis=1)
with open('output/training_data.json', mode='w') as f:
    json.dump(data, f)