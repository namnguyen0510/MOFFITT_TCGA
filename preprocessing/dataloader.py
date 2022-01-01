import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from utils import *
from functools import reduce
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import *
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import *

class TCGADataset(Dataset):
	"""
	Input:  Data directory of TCGA_Processed data
	Output: Tensors
	"""
	def __init__(self, dirc):
		super(TCGADataset,self).__init__()
		self.dirc = [dirc + '/' + x for x in sorted(os.listdir(dirc))]
		self.all_data = sorted(os.listdir(dirc))
		self.inputs = []
		self.keys = []
		for i, k in enumerate(self.dirc):
			print('[Loading Data {:4d}/{:4d}]'.format(i,len(self.dirc)))
			if k != os.path.join(dirc,'label.csv'):
				self.inputs.append(pd.read_csv(k, sep = '\t'))
				self.keys.append(self.all_data[i].split('.')[0])
		self.labels = pd.read_csv(os.path.join(dirc, 'label.csv')).to_numpy().ravel()
		#self.inputs = dict(zip(self.keys,self.inputs))
		label_encoder = preprocessing.LabelEncoder()
		self.labels = label_encoder.fit_transform(self.labels)

	def _split_train_test(self, df, test_size, seed):
		x_train, x_test, y_train, y_test = train_test_split(df, self.labels, test_size = test_size, random_state = seed)
		return x_train, x_test, y_train, y_test


	def _to_tensor(self, df):
		output = []
		for k in self.keys:
			tensor = df[k].to_numpy()
			tensor = torch.tensor(tensor)
			output.append(tensor)
			#print('Tensor Size: {}'.format(tensor.size()))
		return output


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		output = []
		for k in self.keys:
			tensor = self.inputs[k][idx].to_numpy()
			tensor = torch.tensor(tensor)
			output.append(tensor)
		label = self.labels[idx]


		return output, label


































############
