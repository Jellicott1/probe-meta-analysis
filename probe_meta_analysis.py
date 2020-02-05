#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:55:51 2020

@author: je10g15
"""

import pandas as pd
import numpy as np
import random as rnd
import os
import matplotlib.pyplot as plt
import scipy.stats as stat

#%% Data reading functions

def read_probe_list(probe_files, loc):
	try:
		data_list = []
		probe_ind = {}
		ind = 0
		for item in probe_files:
			data_list.append(pd.read_csv('{}{}'.format(loc,item)))
			probe_ind[item.strip('.csv')] = ind
			ind += 1
	except:
		raise TypeError('Invalid Probe ID(s).')
	return data_list, probe_ind

def read_probes(probe_ids, loc:str='Probe-Data/', rand:bool=False):
	if rand == True:
		if isinstance(probe_ids, int):
			files = np.array(os.listdir('{}/{}'.format(os.getcwd(),loc)))
			inds = rnd.sample(range(len(files)),k=probe_ids)
			files = files[inds]
			return read_probe_list(files,loc)
		else:
			raise TypeError('With rand=True, probe_ids should be an int specifying the number of probes to sample.')
	if isinstance(probe_ids, str):
		return pd.read_csv('{}{}.csv'.format(loc,probe_ids))
	elif isinstance(probe_ids, list):
		for i in range(len(probe_ids)):
			probe_ids[i] += '.csv'
		read_probe_list(probe_ids)
	else:
		raise TypeError('Invalid probe_ids type.')

#%%  Plotting functions

def plt_p_vs_adj_p(data, n=None):
	plt.figure()
	if n == None:
		n = len(data)
	elif n > len(data):
		raise ValueError('n cannot be greater than the number of proves in the data.')
	if n > 1 and len(data) > 1:
		fig, ax = plt.subplots(1,n)
		for i in range(n):
			ax[i].plot(data[i].sort_values(by='p_value',ascending=False)['p_value'].values)
			ax[i].scatter(np.arange(data[i].shape[0]),data[i].sort_values(by='p_value',ascending=False)['adj_p_value'].values, s=1, color='orange')
		fig.legend(['p-value','Adj p-value'],loc='upper center')
	elif len(data) > 1:
		plt.plot(data[0].sort_values(by='p_value',ascending=False)['p_value'].values)
		plt.scatter(np.arange(data[0].shape[0]),data[0].sort_values(by='p_value',ascending=False)['adj_p_value'].values, s=1, color='orange')
	elif n == 1:
		plt.plot(data.sort_values(by='p_value',ascending=False)['p_value'].values)
		plt.scatter(np.arange(data.shape[0]),data.sort_values(by='p_value',ascending=False)['adj_p_value'].values, s=1, color='orange')
	else:
		raise ValueError('n cannot be smaller than 1')
	plt.show()

def plt_probes_per_block_hist(block_data, bins=None):
	plt.figure()
	plt.hist(block_data['Unique Blocks'],bins=bins)
	plt.ylabel('Frequency')
	plt.xlabel('Number of Probes')
	plt.title('Histogram of Probe Block Counts')
	plt.show()

def plt_many_probes(data, col='p_value'):
	plt.figure()
	for item in data:
		plt.plot(item.sort_values(by=col,ascending=False)[col].values)
	plt.ylabel(col)
	plt.xlabel('Sample Ranking')
	plt.title('Superimposed plots of {} for many probes'.format(col))
	plt.show()

#%% Read data

data,key = read_probes(1000, rand=True)
block_data = pd.read_csv('probe_block_counts.csv')

#%%

plt_many_probes(data)

block_data.describe()
stat.iqr(block_data['Unique Blocks'])
stat.tstd(block_data['Unique Blocks'])

# block_probe_counts = pd.read_csv('block_probe_counts.csv',index_col=0)
# Nblocks = block_probe_counts.shape[0]
# low_count_blocks = block_probe_counts[block_probe_counts['Unique Probes'] < block_probe_counts.describe().iloc[4,0]]
# low_count_blocks = block_probe_counts.sort_values('Unique Probes').iloc[0:300].index
# block_data = block_data[block_data['']]
