import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
import os
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.feature_selection import *#VarianceThreshold, SelectKBest,chi2
#import plotly as plt
#import plotly.express as px
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		param = parameter.numel()
		table.add_row([name, param])
		total_params+=param
	print(table)
	print(f"Total Trainable Params: {total_params/1000000} M")
	return total_params

class FeatureExtractor():
    """
    Input:  Data directory of RAW data
    Output: Data directory of PROCESSING data
    """
    def __init__(self, dirc):
        super(FeatureExtractor,self).__init__()
        self.omics = ['Clinical_Clinical_Clinical',
                    'miRNASeq_HS_miR_Gene',
                    'Methylation_Meth450_Gene',
                    'RNAseq_HiSeq_RNA_Gene',
                    'Mutation_GAIIx_Gene']
        self.dirc = dirc


        self.out_dims = []



        self.all_omics = sorted(os.listdir(dirc))
        self.DFs = []
        self.omic_names = []
        for o in self.all_omics:
            name = o.split('__')[3] + '_' + o.split('__')[4] + '_' + o.split('__')[7]
            #print(name)
            if name in self.omics:
                #print(o)
                df = pd.read_csv(os.path.join(dirc, o), sep = '\t')
                self.DFs.append(df)
                if 'Clinical' in name.split('_'):
                    self.omic_names.append('Clinical')
                else:
                    self.omic_names.append(name.split('_')[0]+'_'+name.split('_')[2])
        self.meta_data = dict(zip(self.omic_names, self.DFs))
        self.all_patients = self._get_patients()
        self.attributes = self._get_attributes()
        self.DFs_clean = self._preprocessing_omic()
        self.meta_data_clean = dict(zip(self.omic_names, self.DFs_clean))
        self.omic_variances = self._get_variances()
        self.reduced_DFs, _omics = self._variance_threshold()
        self.get_raw_data, _ = self._get_raw_data()
        #out_dims = [128,1024,1024,64]
        #args = dict(zip(_omics, out_dims))
        #self.final_DFs = self._select(args)


    ## Info. Retrival
    def _get_dict(self):
        return self.meta_data

    def summary(self, df):
        for i in range(len(df)):
            omic_name = self.omic_names[i]
            data = df[i]
            num_missing = data.isna().sum().sum()
            print('Omic Name: {:16}|Number of Patients: {:6d}|Number of Features: {:6d}|Missing Values: {:6d}'.format(
                    omic_name,  len(data.columns), len(data), num_missing))

    def _summary(self, df):
        for i in range(len(df)):
            omic_name = self.omic_names[i]
            data = df[i]
            num_missing = data.isna().sum().sum()
            print('|{:16}|N: {:6d}|D: {:6d}|'.format(
                    omic_name, len(data.columns), len(data)))

    def _Summary(self, df):
        for i in range(len(df)):
            if self.omic_names[i] != 'Clinical':
                omic_name = self.omic_names[i]
                data = df[i]
                num_missing = data.isna().sum().sum()
                print('|{:16}|N: {:6d}|D: {:6d}|'.format(
                        omic_name, len(data), len(data.columns)))


    def _get_attributes(self):
        df = []
        for omic in self.DFs:
            df.append(omic.iloc[:,0].to_numpy())
        print('# Attributes: {}'.format(len(df)))
        return df

    def _get_patients(self):
        df = []
        for omic in self.DFs:
            df.append(omic.columns[1:].to_numpy())
        df = reduce(np.intersect1d, df)
        print('# Patients: {}'.format(len(df)))
        return df

    def _save(self, dfs, dirc):
        for i, omic in enumerate(dfs):
            omic_name = self.omic_names[i]
            omic.to_csv(os.path.join(dirc, omic_name), sep = '\t', index = False)


    def _preprocessing_omic(self, fillna = 'mean'):
        df = []
        for i, omic in enumerate(self.DFs):
            data = omic[self.all_patients]
            if fillna == 'zero':
                data = data.fillna(0)
                data = (data - data.min())/(data.max()-data.min())
            elif fillna == 'mean':
                data = data.fillna(data.mean())
                data = (data - data.min())/(data.max()-data.min())
            else:
                print('Only support fill NA: zero and mean')

            data.insert(0, 'attribute', self.attributes[i])
            df.append(data)
        return df


    def _get_raw_data(self):
        df = []
        omic_names = []
        for i, omic in enumerate(self.DFs_clean):
            if self.omic_names[i] not in ['Clinical']:
                t = self.omic_variances[self.omic_names[i]]
                features = omic['attribute']
                omic = omic.iloc[:,1:]
                data = omic.T
                data = pd.DataFrame(data)
                data.columns = features
                df.append(data)
                omic_names.append(self.omic_names[i])
            else:
                ## To-to: clinical data is not extract
                data = omic.T.iloc[1:,:]
                #print(data)
                df.append(data)


        return df, omic_names

    """
    def _select(self, args):
        df = []
        for i, omic in enumerate(self.reduced_DFs):
            if self.omic_names[i] !='Clinical':
                out_dim = args[self.omic_names[i]]
                data = omic.iloc[:,:out_dim]
                df.append(data)
            else:
                df.append(omic)
        return df
    """
























class FilteringFeature():
	"""
	Input:  Data directory of PROCESSED data
	Output: Data directory of MULTI-OMIC data
	"""
	def __init__(self, dirc):
		super(FilteringFeature,self).__init__()
		self.cancer_types = sorted(os.listdir(dirc))
		self.omic_names = sorted(os.listdir(os.path.join(dirc, self.cancer_types[0])))
		#self.clinical = []
		self.methyl = []
		self.miRNA = []
		#self.mutation = []
		self.RNAseq = []
		for type in self.cancer_types:
			for omic in self.omic_names:
				data = pd.read_csv(os.path.join(dirc, os.path.join(type, os.path.join(omic))), sep = '\t')
				#if omic == "Clinical":
					#self.clinical.append(data)
				if omic == "Methylation_Gene":
					self.methyl.append(data)
				elif omic == "miRNASeq_miR":
					self.miRNA.append(data)
				#elif omic == "Mutation_Gene":
					#self.mutation.append(data)
				elif omic == "RNAseq_RNA":
					self.RNAseq.append(data)
				#else:
					#print('Find wrong Query in folder: {}'.format(os.path.join(dirc, os.path.join(type, os.path.join(omic)))))

					#print("|{:8}|{:16}|N: {:6d}|D: {:6d}|".format(type,omic,n,d))
		self.keys = ["Methyl","miRNA","RNAseq"]
		self.data = [self.methyl, self.miRNA, self.RNAseq]
		self.meta_data = dict(zip(self.keys,self.data))
		#print(len(self.meta_data["Methyl"]))
		self.group_by_omic, self.all_features = self.get_meta()

		self.common_feature = self._get_common_feat()
		var_thres = dict(zip(self.keys,[0.03, 0.0005, 0.01]))
		self._x, self._y = self._filter(var_thres)

		print(self._y)
		self.out_dirc = 'omic_data'





	def _filter(self, var_thres):
		dfs = []
		for k in self.keys:
			data = self.group_by_omic[k]
			f = self.common_feature[k]
			#print('-----------------------------------------')
			df = []
			labels = []
			fs = []
			for i, d in enumerate(data):
				data = d[f]
				#features = data.columns
				#var = data.var(axis = 0).mean()
				#selector = VarianceThreshold(var_thres[k])
				#selector.fit(data)
				#mask = selector.get_support()
				#selected_features = features[mask]
				#data = data[selected_features]
				#print(data)
				df.append(data)
				#print('|{:16}|N: {:8d}|D: {:8d}'.format(k, len(data), len(data.columns)))
				fs.append(data.columns)
				#labels.append(self.cancer_types[i])
				label = [self.cancer_types[i]]*len(data)
				labels.append(np.array(label,dtype = object))
			df = pd.concat(df, axis = 0)
			dfs.append(df)
		labels = np.hstack(labels)
		l = pd.DataFrame([])
		l['label'] = labels
		l.to_csv(os.path.join('omic_data','label.csv'), index = False)
		for i, d in enumerate(dfs):
			d.to_csv(os.path.join('omic_data', '{}.csv'.format(self.keys[i])), sep = '\t', index = False)
			print('|{:16}|N: {:8d}|D: {:8d}'.format(self.keys[i], len(d), len(d.columns)))

		return dict(zip(self.keys,dfs)), labels


	def _get_common_feat(self):
		out = []
		for omic in self.all_features:
			feat = []
			for f in omic:
				feat.append(f)
			feat = reduce(np.intersect1d, feat)
			out.append(feat.tolist())
		return dict(zip(self.keys, out))


	def _concat(self):
		v = []
		for i, omics in enumerate(self.group_by_omic):
			value = self.group_by_omic[omics]
			v.append(value)
		v = torch.cat(v, dim = 1)
		#print(v.size())
		return v


	def get_meta(self):
		df = []
		k = []
		f = []
		for key in self.keys:
			c, feat = self._group(key)
			k.append(key)
			df.append(c)
			f.append(feat)
		return dict(zip(k, df)), f

	def _group(self, key):
		df = []
		feat = []
		output = self.meta_data[key]
		for omic in output:
			#print("|{:16}|N: {:6d}|D: {:6d}|".format(key, omic.shape[0], omic.shape[1]))
			df.append(omic)
			feat.append(omic.columns)
		return df, feat







class ReduceDim():
	"""
	Input:  Data directory of Mult-Omic data
	Output: Data directory of Mult-Omic data with Reduced Dimensionality
	"""
	def __init__(self, dirc):
		super(ReduceDim,self).__init__()
		self.omics = [dirc + '/' + x for x in sorted(os.listdir(dirc))]
		self.keys = [x.split('/')[1].split('.')[0] for x in self.omics]
		self.values = []
		for omic in tqdm.tqdm(self.omics):
			self.values.append(pd.read_csv(omic, sep = '\t'))
		var = [0.005, 0.0008, 0.0001]
		self.feat_variance = dict(zip(set(self.keys) - set(['label']),var))
		self.reduced_data = self._variance_threshold()
		self.labels = pd.read_csv(os.path.join(dirc,'label.csv'))
		num_feats = [8196, 512, 8196]
		self.num_feats = dict(zip(set(self.keys) - set(['label']),num_feats))
		self._out = self._select_k_best()

	def _select_k_best(self, type = 'chi2'):
		df = []
		print('-Feature Selecting')
		for i, omic in enumerate(self.values):
			if self.keys[i] != 'label':
				data = omic
				"""
				keys = data.columns
				selector = SelectKBest(chi2, k = self.num_feats[self.keys[i]])
				selector.fit(omic, self.labels)
				mask = selector.get_support()
				data = omic.loc[:,mask]
				"""
				df.append(data)
				data.to_csv(os.path.join('TCGA_Processed','{}.csv'.format(self.keys[i])), sep = '\t', index = False)
				print('|{:16}|N: {:8d}|D: {:8d}'.format(self.keys[i], len(data), len(data.columns)))
			else:
				data = omic
				df.append(omic)
				data.to_csv(os.path.join('TCGA_Processed','{}.csv'.format(self.keys[i])), sep = '\t', index = False)
		return df



	def _variance_threshold(self, save = True):
		df = []
		print('-Variance Thresholding')
		for i, omic in enumerate(self.values):
			if self.keys[i] not in ['label','miRNA']:
				selector = VarianceThreshold(self.feat_variance[self.keys[i]])
				selector.fit(omic)
				mask = selector.get_support()
				#print(data.shape)
				#print(len(mask))
				data = omic.loc[:,mask]
				df.append(data)
				if save:
					data.to_csv(os.path.join('var_reduced_data','{}.csv'.format(self.keys[i])), sep = '\t', index = False)
				print('|{:16}|N: {:8d}|D: {:8d}'.format(self.keys[i], len(data), len(data.columns)))
			else:
				data = omic
				if save:
					data.to_csv(os.path.join('var_reduced_data','{}.csv'.format(self.keys[i])), sep = '\t', index = False)
				df.append(data)
		return df


	def _get_variance(self, reduce = 'mean'):
		vars = []
		for i, omic in enumerate(self.values):
			if self.keys[i] != 'label':
				if reduce == 'mean':
					v = omic.var(axis = 1).mean()
					vars.append(v)
				else:
					v = omic.var(axis = 1)
					vars.append(v)
			else:
				vars.append(-1)
		return dict(zip(self.keys,vars))



























####
