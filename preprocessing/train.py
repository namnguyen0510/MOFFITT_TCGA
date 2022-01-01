import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from dataloader import *
from utils import *
from model import *
import seaborn as sn
"""
This code is for TRAIN model on TCGA data
- ARGS: seed, epochs, INDEX of TCGA data
-- INDEX 0: Methyl
-- INDEX 1: RNAseq
-- INDEX 2: miRNA
"""
def main(seed, epochs, idx):

    seed = 7
    #epochs = 2000
    #idx = 2

    dirc = 'TCGA_Processed'
    data = TCGADataset(dirc)
    print(data.keys)

    omic = data.inputs[idx]
    print('--Training: {:8}'.format(data.keys[idx]))
    x_train, x_test, y_train, y_test = data._split_train_test(omic, test_size = 0.4, seed = seed)
    _data = [x_train, x_test, y_train, y_test]
    _name = ['x_train', 'x_test', 'y_train', 'y_test']

    """
    # Save folds
    for i, d in enumerate(_data):
        pd.DataFrame(d).to_csv(_name[i]+'.csv',index = False)
    """
    x_train = torch.tensor(x_train.to_numpy())
    y_train = torch.tensor(y_train)

    x_test = torch.tensor(x_test.to_numpy())
    y_test = torch.tensor(y_test)

    #print('Train Data: {}'.format(x_train.size()))
    #print('Test Data: {}'.format(x_test.size()))
    if idx == 2:
        in_dim = 517
    else: in_dim = 19416
    model = MOSANet(in_dim = in_dim, num_classes = 16).cuda()
    count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    for epoch in tqdm.tqdm(range(epochs)):
        train_loss, train_acc, train_report = train(x_train, y_train, model, criterion, optimizer, scheduler, epoch)
        test_loss, test_acc, test_report = eval(x_test, y_test, model, criterion, epoch)
        #print('Epoch: {:6d}|TRAIN|LR: {:4f}|Loss: {:6f}|Accuracy: {:6f}'.format(epoch, scheduler.get_last_lr()[0], train_loss, train_acc))
        #print('Epoch: {:6d}|TEST |LR: {:4f}|Loss: {:6f}|Accuracy: {:6f}'.format(epoch, scheduler.get_last_lr()[0], test_loss, test_acc))

    print('---Train Acc: {:8f}'.format(train_acc))
    print('---Test Acc: {:8f}'.format(test_acc))
    df_cm = pd.DataFrame(test_report)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    #plt.show()


def train(sample, target, model, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0
    n = 0
    y_true = []
    y_pred = []
    model.zero_grad()
    optimizer.zero_grad()
    sample = sample.cuda()
    target = target.cuda()
    logits = model(sample)
    loss = criterion(logits, target)
    logits = torch.argmax(logits,dim=1)
    #print(target)
    #print(logits)
    loss.backward()
    optimizer.step()
    scheduler.step()
    running_loss += loss.item()
    n += target.size(0)
    y_true.append(target.detach().cpu().numpy())
    y_pred.append(logits.detach().cpu().numpy())
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    df = pd.DataFrame([])
    df['y_true'] = y_true.astype('int')
    df['y_pred'] = y_pred#.astype('int')
    accuracy = accuracy_score(df['y_true'], df['y_pred'])
    report = classification_report(df['y_true'], df['y_pred'])
    try:
        os.mkdir('./results')
    except:
        pass
    #df.to_csv(os.path.join('./results', 'result_{}.csv'.format(epoch)), index = False, sep = '\t')
    return running_loss/n, accuracy, report

def eval(sample, target, model, criterion, epoch):
    model.eval()
    #running_loss = 0
    n = 0
    y_true = []
    y_pred = []
    sample = sample.cuda()
    target = target.cuda()
    logits = model(sample)
    loss = criterion(logits, target)
    logits = torch.argmax(logits,dim=1)
    #running_loss += loss.item()
    loss = loss.detach().cpu().numpy()
    n += target.size(0)
    y_true.append(target.detach().cpu().numpy())
    y_pred.append(logits.detach().cpu().numpy())
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    df = pd.DataFrame([])
    df['y_true'] = y_true.astype('int')
    df['y_pred'] = y_pred#.astype('int')
    accuracy = accuracy_score(df['y_true'], df['y_pred'])
    #report = classification_report(df['y_true'], df['y_pred'])
    report = confusion_matrix(df['y_true'], df['y_pred'], normalize='true')
    try:
        os.mkdir('./results')
    except:
        pass
    df.to_csv(os.path.join('./results', 'result_{}.csv'.format(epoch)), index = False, sep = '\t')
    return loss/n, accuracy, report
