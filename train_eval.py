# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
import numpy as np
import random

#model training
def training(train_iter, dev_iter, model, args, device):
    l2 = args.l2
    # move model to device before constructing optimizer for it.
    model.to(device)
    if not args.static and not args.sim_static:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2)
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=l2)

    model_path = './best_model'
    total_step = len(train_iter) #num of steps in each epoch
    train_accs = [] #train data accuracy
    dev_accs = [] #valid data accuracy
    best_acc = 0 #best accuracy
    exp_acc = 0.96 #expected accuracy
    t0 = time() #record the beginning time

    for epoch in range(1, args.epochs + 1):
        model.train()  # training mode, we should reset it to training mode in each epoch
        for i, batch in enumerate(train_iter):
            texts, labels = batch.text.to(device), batch.label.to(device) - 1
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Clears the gradients
            # Visualization of the train process
            print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, args.epochs, i+1, total_step, loss.item()), end='', flush=True)
        print(end='\n')
        # in each epoch we call eval(), switch to evaluation mode
        train_acc = evaluating(train_iter, model, device)
        train_accs.append(train_acc)
        dev_acc = evaluating(dev_iter, model, device)
        dev_accs.append(dev_acc)
        print('train_acc: %.1f%%, valid_acc:%.1f%%' % (train_acc * 100, dev_acc * 100))
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model, model_path) #save the best model
        if best_acc > exp_acc:
            print('Reach to accuracy expected, early stopping')
            break

    t1 = time() #record end time
    print('Total training time: %.2f' % (t1 - t0))
    show_training(train_accs, dev_accs)
    return model

# evaluate the accuracy for (data_iter, model, device)
def evaluating(data_iter, model, device):
    model.to(device)
    model.eval()  # evaluation mode
    with torch.no_grad():
        correct, avg_loss = 0, 0
        for batch in data_iter:
            texts, labels = batch.text.to(device), batch.label.to(device) - 1
            # print(texts)
            outputs = model(texts)
            predicted = torch.max(outputs.data, 1)[1]
            loss = F.cross_entropy(outputs, labels, reduction='mean')

            avg_loss += loss.item()
            correct += (predicted == labels).sum()

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = correct.item() / size  
        return accuracy

#show training,validing accuracy using pyplot as .png
def show_training(train_accs, dev_accs):
    # plot train acc and validation acc
    filePath = './datafig.png'
    plt.figure()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('train_acc and dev_acc v.s. epochs')
    plt.tight_layout()
    # plt.xticks(range(0,args.epochs),range(1,args.epochs+1))
    plt.plot(train_accs, label='train_acc')
    plt.plot(dev_accs, label='dev_acc')
    plt.legend()
    plt.savefig(filePath)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False