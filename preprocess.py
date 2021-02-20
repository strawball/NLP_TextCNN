# -*- coding: utf-8 -*-

import pandas as pd
from nltk import word_tokenize
import os

task = './data'
data_source = 'dbpedia' #data source
#data_path:the path to where data chosen places
data_path = os.path.join(task, data_source)
#raw_path:the path to where raw data places
raw_path = os.path.join(task, data_source, 'raw')
#load data(translate from .csv to dataset)
full = pd.read_csv(os.path.join(raw_path, 'train.csv'), header=None, sep=',') # all data for training and validing
test = pd.read_csv(os.path.join(raw_path, 'test.csv'), header=None, sep=',') # data for testing
#concatenate all columns except the label column
def preprocess(dataset):
    dataset.fillna('', inplace=True)
    dataset['text'] = dataset[1].str.cat(
        [dataset[i] for i in range(2, dataset.shape[1])], sep=' ')
    dataset.drop(columns=range(1, dataset.shape[1] - 1), inplace=True)
    return dataset
#concatenation on training/testing file
full = preprocess(full)
test = preprocess(test)
full.rename(columns={0: 'label'}, inplace=True)
test.rename(columns={0: 'label'}, inplace=True)

#split train and test data (or use k-fold Cross-validation)
train = full.sample(frac=0.9, random_state=0, axis=0)
dev = full[~full.index.isin(train.index)]

train.to_csv(os.path.join(data_path, 'train.csv'), header=True, index=False)
dev.to_csv(os.path.join(data_path, 'dev.csv'), header=True, index=False)
test.to_csv(os.path.join(data_path, 'test.csv'), header=True, index=False)

def get_length(dataset):
    return dataset.shape[0]