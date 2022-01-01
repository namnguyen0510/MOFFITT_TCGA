import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from dataloader import *
import os
from model0 import *
from engine import *
from capsnet import *

seed = 42

dirc = '../TCGA_Processed_2048'
omic_type = 'RNAseq'
test_size = 0.2
data_description = True
report = True
print_max_prob = False

epochs = 10000

log_freq = 2
if torch.cuda.is_available():
	device = 'cuda:0'


'''Load SINGLE Omic data: Default: RNAseq ['Methyl', 'miRNA']'''
omic = TCGASingleOmic(dirc, type = omic_type)
data = omic.data
label = omic.label
x_train, x_test, y_train, y_test = omic._split_train_test(data, test_size = test_size, seed = seed)
_data = [x_train, y_train, x_test, y_test]
_data = [torch.tensor(d) for d in _data]
_name = ['x_train', 'y_train', 'x_test', 'y_test']

'''Each fold will be a DICT {'dataqueue', 'tensor'}'''
#print(_fold)
_fold = dict(zip(_name,_data))
if data_description:
	print('----- Description-----')
	print('----- {:6}'.format(omic_type))
	print('#Features | {:10d}'.format(_data[0].size(1)))
	for i, d in enumerate(_data):
		print("{:10}|N: {:8d}".format(_name[i], d.size(0)))

'''Complile Model'''
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr = 3e-1, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))
if data_description:
	get_model_size(model)


'''Evaluate Fold'''

evaluator = EvalFold(_fold, model, criterion, optimizer, scheduler, epochs)
#print(evaluator.train())
#print(evaluator.eval())
for epoch in range(epochs):
	evaluator.step()
	if report:
		if (epoch > 0) and (epoch%log_freq == 0):
			print('------')
			print('LR: {:4f}'.format(scheduler.get_last_lr()[0]))
			print('Epoch: {:4d}|TRAIN|Loss: {:6f}|Accuracy: {:6f}'.format(epoch, evaluator.train_loss[-1], evaluator.train_acc[-1]))
			print('Epoch: {:4d}|TEST |Loss: {:6f}|Accuracy: {:6f}'.format(epoch, evaluator.test_loss[-1], evaluator.test_acc[-1]))
			print('Train Score: {:4f}'.format(max(evaluator.train_acc)))
			print('Test  Score: {:4f}'.format(max(evaluator.test_acc)))
			if print_max_prob:
				for i in range(model._get_prob()[0]):
					print('Min Prob.: {:4f}, Max Prob.: {:4f}'.format(model._get_prob()[1][i].min(),model._get_prob()[1][i].max()))















#########3
