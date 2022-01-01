import os
import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from functools import reduce
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import *

class TCGASingleOmic(Dataset):
	def __init__(self, dirc, type = 'RNAseq'):
		super(TCGASingleOmic,self)
		self.dirc = [dirc + '/' + x for x in sorted(os.listdir(dirc))]
		self.all_data = sorted(os.listdir(dirc))
		self.omic_type = '{}.csv'.format(type)
		self.omic_index = self.all_data.index(self.omic_type)
		self.data = pd.read_csv(self.dirc[self.omic_index], sep = '\t')
		self.keys = self.data.columns
		self.data = self.data.to_numpy()
		self.label = pd.read_csv(os.path.join(dirc, 'label.csv')).to_numpy().ravel()
		label_encoder = preprocessing.LabelEncoder()
		self.label = label_encoder.fit_transform(self.label)

	def _split_train_test(self, df, test_size, seed):
		x_train, x_test, y_train, y_test = train_test_split(df, self.label, test_size = test_size, random_state = seed, stratify = self.label)
		return x_train, x_test, y_train, y_test


	def _to_tensor(self, df):
		tensor = df.to_numpy()
		tensor = torch.tensor(tensor)
		return tensor
