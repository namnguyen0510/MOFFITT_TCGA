import torch
import torch.nn as nn
from utils import *

'''
Class funcfion for Evaluate each Fold
Input:  Fold, ARGs
output: Train-Test ACC and LOSS
'''
class EvalFold():
	def __init__(self, fold, model, criterion, optimizer, scheduler, epochs):
		super(EvalFold, self)
		device = 'cuda'
		self.epochs = epochs
		self.fold = fold
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler


		self.x_train = fold['x_train'].to(device)
		self.y_train = fold['y_train'].to(device)
		self.x_test = fold['x_test'].to(device)
		self.y_test = fold['y_test'].to(device)
		self.n_train = self.y_train.size(0)
		self.n_test = self.y_test.size(0)

		self.train_loss = []
		self.train_acc = []
		self.test_loss = []
		self.test_acc = []



	def step(self):
		train_loss, train_acc = self.train()
		self.train_loss.append(train_loss)
		self.train_acc.append(train_acc)
		test_loss, test_acc = self.test()
		self.test_loss.append(test_loss)
		self.test_acc.append(test_acc)

	def train(self, report = False, log_freq = 100):
		self.model.train()
		'''Will be useful for minibatch training'''
		for epoch in range(1):
			objs = AvgrageMeter()
			score = AvgrageMeter()
			self.model.zero_grad()
			self.optimizer.zero_grad()
			logits = self.model(self.x_train)
			loss = self.criterion(logits, self.y_train)
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()
			acc, _ = accuracy(logits, self.y_train, topk=(1, 5))
			objs.update(loss.item(), self.n_train)
			score.update(acc.item(), self.n_train)
			if report:
				if (epoch > 0) and (epoch%log_freq == 0):
					print('Epoch: {:4d}|TRAIN|Loss: {:8f}|Accuracy: {:8f}'.format(epoch, objs.avg, score.avg))
		return objs.avg, score.avg

	def test(self, report = False, log_freq = 100):
		self.model.eval()
		for epoch in range(1):
			objs = AvgrageMeter()
			score = AvgrageMeter()
			logits = self.model(self.x_test)
			loss = self.criterion(logits, self.y_test)
			acc, _ = accuracy(logits, self.y_test, topk=(1, 5))
			objs.update(loss.item(), self.n_test)
			score.update(acc.item(), self.n_test)
			if report:
				if (epoch > 0) and (epoch%log_freq == 0):
					print('Epoch: {:4d}|Test|Loss: {:8f}|Accuracy: {:8f}'.format(epoch, objs.avg, score.avg))
		return objs.avg, score.avg























########
