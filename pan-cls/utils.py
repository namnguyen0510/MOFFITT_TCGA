import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from prettytable import PrettyTable

def get_model_size(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		param = parameter.numel()
		table.add_row([name, param])
		total_params+=param
	print(table)
	print("#Params {:10f} M".format(total_params/1000000))
	return total_params

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res

class AvgrageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


	def accuracy(output, target, topk=(1,)):
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.reshape(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0)
			res.append(correct_k.mul_(100.0/batch_size))
		return res
