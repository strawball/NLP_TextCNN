# -*- coding: utf-8 -*-

import torch
import os
from torchtext import data
from torchtext.vocab import Vectors
from nltk.tokenize import word_tokenize
from models import TextCNN 
from settings import Settings
from train_eval import training
import pickle

data_source = 'dbpedia'
models = {
    'TextCNN': TextCNN,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = './data'
vector_path = os.path.join(data_path, 'vectors')
TEXT_path = os.path.join(vector_path, 'TEXT.pkl')
dataset_path = os.path.join(data_path, data_source)
class_vecs = torch.randn(14, 300)  #14/dbpedia, 10/yahoo
length = 100  #100/dbpedia; 300/yahoo

#set batch_first=True
TEXT = data.Field(
    sequential=True,
    tokenize=word_tokenize,
    lower=True,
    fix_length=length,  #according to the dataset
    batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
fields = {'label': LABEL, 'text': TEXT}
#bulid cache
cache = os.path.join(vector_path, 'vector_cache')
if not os.path.exists(cache):
    os.mkdir(cache)
#use Glove embeddings to build vector
vectors = Vectors(
    name=os.path.join(vector_path, 'glove.6B.300d.txt'),#torch.Size([2196017, 300])
    cache=cache)
#load data set
train, dev, test = data.TabularDataset.splits(
    path=dataset_path,
    train='train.csv',
    validation='dev.csv',
    test='test.csv',
    format='csv',
    fields=[('label', LABEL), ('text', TEXT)],
    skip_header=True)
#construct the vocab, filter low frequency words if needed
TEXT.build_vocab(train, min_freq=2, vectors=vectors)
#dump the vocab 
with open(TEXT_path, 'wb') as f:
    pickle.dump(TEXT, f)
del vectors  #del vectors to save space

if __name__ == '__main__':
    #settings, see settings.py for detail
    label_vecs = class_vecs.unsqueeze(1).unsqueeze(2)  #u can ignore this
    args = Settings(
        TEXT.vocab.vectors,  #pre-trained word embeddings
        label_vecs,
        L=length,
        Dim=300,             #embedding dimension
        num_class=14,        #14/dbpedia, 10/yahoo
        Cout=256,            #kernel numbers
        kernel_size=[2, 3, 4],  #different kernel size
        dropout=0.5,
        num_epochs=10,
        lr=0.001,
        weight_decay=0,
        static=True,            #update the embeddings or not
        sim_static=True,    #only used for sims.py: update label_vec_kernel or not
        batch_size=256,
        batch_normalization=False,
        hidden_size=256,    #100,128,256...
        bidirectional=True)  #birnn/rnn
    #construct dataset iterator
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test),
        sort_key=lambda x: len(x.text),
        batch_sizes=[args.batch_size] * 3,
        device = torch.device('cuda:0')
    )

    classifier = 'TextCNN'
    if args.static:
        print('static %s(without updating embeddings):' % classifier)
    else:
        print('non-static %s(update embeddings):' % classifier)
    model = models[classifier](args)
    #train the model
    training(train_iter, dev_iter, model, args, device)
    print('Training Ends')