# -*- coding: utf-8 -*-

#evaluate the test dataset for the model
import os
import pickle
from torchtext import data
import torch
from train_eval import evaluating
from time import time

data_source = 'dbpedia'
datatest_path = os.path.join('./data', data_source, 'test.csv')
model_path = './best_model'
TEXT_path = './data/vectors/TEXT.pkl'
# loda the TEXT
with open(TEXT_path, 'rb') as f:
    TEXT = pickle.load(f)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
model = torch.load(model_path)
model.to(torch.device('cpu'))
# 
#evaluate the test set
def test_dataset_eval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        print('There is not model trained,please train the model first.')
    else:
        #construct test dataset
        test = data.TabularDataset(
            path=datatest_path,
            format='csv',
            fields=[('label', LABEL), ('text', TEXT)],
            skip_header=True)
        #construct test dataset iterator
        test_iter = data.BucketIterator(
            test,
            batch_size=256,
            sort_key=lambda x: len(x.text),
            device = torch.device('cuda:0'),
            train=False)
        t0 = time()
        test_acc = evaluating(test_iter, model, device)
        t1 = time()
        #get test length, test accuracy and testing time
        print('test data counts: %d' % (len(test_iter.dataset)))
        print('test_acc: %.1f%%' % (test_acc * 100)) #98.8/dbpedia
        print('testing time: %.2f' % (t1 - t0))
#test for just one sentence
def test_sentense_pred(st):
    st =  TEXT.preprocess(st)
    s = [[TEXT.vocab.stoi[x] for x in st]]
    s = torch.tensor(s)
    out = model(s)
    return torch.max(out, 1)[1]

if __name__ == '__main__':
    opt = input('Please choose the mode: \'1\' for using test dataset to test;\'2\' for inputing a sentence to test \n')
    if opt == '1':
        test_dataset_eval()
    else:
        str_test = input('Please inputing the sentence for testing:\n')
        cat = test_sentense_pred(str_test)
        print('category predicted: ' + str(cat[1]))