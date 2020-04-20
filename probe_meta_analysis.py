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
from scipy.signal import savgol_filter
from lmfit import Model
import pickle, json

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
	if probe_ids == 'all':
		files = np.array(os.listdir('{}/{}'.format(os.getcwd(),loc)))
		return read_probe_list(files,loc)
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

def read_probes_all(loc:str='Probe-Data/'):
	files = np.array(os.listdir('{}/{}'.format(os.getcwd(),loc)))
	return read_probe_list(files, loc)

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
	
def plt_many_probes(data, col='p_value', log=False):
	plt.figure()
	if log == False:
		for item in data:
			plt.plot(np.linspace(1,0,len(item)),item.sort_values(by=col,ascending=False)[col].values)
		plt.title('Superimposed plots of {} for many probes'.format(col))
		plt.ylabel(col)
	else:
		for item in data:
			plt.plot(np.linspace(1,0,len(item)),np.log10(item.sort_values(by=col,ascending=False)[col].values))
		plt.title('Superimposed plots of log {} for many probes'.format(col))
		plt.ylabel('log {}'.format(col))
	plt.xlabel('Sample Ranking')
	plt.show()

def plt_poly_vs_weib_fit(data, n, weight='tail'):
	fig, ax = plt.subplots(1,n)
	fig2, ax2 = plt.subplots(1,2)
	ax2[0].title.set_text('Exp-Weibull')
	ax2[1].title.set_text('Poly')
	for i in range(n):
		leg = ['Data {}'.format(i)]
		if weight == 'symmetric':
			if len(data[i])%2 == 0:
				weights = np.concatenate((np.logspace(1,0,int(len(data[i])/2)),np.logspace(0,1,int(len(data[i])/2))))
			else:
				weights = np.concatenate((np.logspace(1,0,int((len(data[i])+1)/2)),np.logspace(0,1,int((len(data[i])-1)/2))))
			x, y, test = model_fit_weib(data, i, weight=weights)
		else:
			x, y, test = model_fit_weib(data, i)
		ax[i].plot(x,test)
		ax[i].plot(x,y)
		ax2[0].plot(x,y)
		leg.append('\nexpweib {}\ncorr = {:.4f}\nerror = {:.2f}'.format(i, np.corrcoef(y,test)[0,1], sum(abs(y-x))))
		if weight == 'symmetric':
			x, y, test = model_fit_poly(data, i, weight=weights)
		else:
			x, y, test = model_fit_poly(data, i)
		ax[i].plot(x,y)
		leg.append('\npoly {}\ncorr = {:.4f}\nerror = {:.2f}'.format(i, np.corrcoef(y,test)[0,1], sum(abs(y-x))))
		ax[i].legend(leg)
		ax2[1].plot(x,y)

#%% Fitting functions
def poly(x, x_2, x_3, x_4):
	return x_2*x**2 + x_3*x**3 + x_4*x**4

def dict_poly(x, coefs):
	return poly(x, coefs['x_2'], coefs['x_3'], coefs['x_4'])

def expweib(x,k,lamda,alpha):
      return alpha*(k/lamda)*((x/lamda)**(k-1))*((1-np.exp(-(x/lamda)*k))**(alpha-1))*np.exp(-(x/lamda)*k)

def expweib_cdf(x,k,l,a):
	return (1-np.exp(-(x/l)**k))**a

def model_fit_poly(data, data_index, weight='default', results=False):
	raw = data[data_index]['p_value'].sort_values(ascending=True).values
	raw = raw[~np.isnan(raw)]
	x = np.linspace(0,1,len(raw))
	pmodel = Model(poly)
	params = pmodel.make_params(x_2=1, x_3=0, x_4=1)
	if weight == 'default':
		result = pmodel.fit(raw, params, x=x, weights=np.logspace(1,0,len(raw)))
	elif weight == None:
		result = pmodel.fit(raw, params, x=x)
	else:
		result = pmodel.fit(raw, params, x=x, weights=weight)
	y = result.eval(x=x)
	if results == True:
		return x, y, result
	else:
		return x, y, raw

def model_fit_weib(data, data_index, weight='default'):
	raw = data[data_index]['p_value'].sort_values(ascending=True).values
	raw = raw[~np.isnan(raw)]
	x = np.linspace(0,1,len(raw))
	pmodel = Model(expweib_cdf)
	params = pmodel.make_params(k=1,l=1,a=1)
	params['k'].min = 0.00000000001
	params['l'].min = 0.00000000001
	params['a'].min = 0.00000000001
	if weight == 'default':
		result = pmodel.fit(raw, params, x=x, weights=np.logspace(1,0,len(raw)))
	elif weight == None:
		result = pmodel.fit(raw, params, x=x)
	else:
		result = pmodel.fit(raw, params, x=x, weights=weight)
	y = result.eval(x=x)
	return x, y, raw

def poly_grad(x, probe_id, coef_dict):
	x_2, x_3, x_4 = coef_dict[probe_id].values()
	return 2*x_2*x + 3*x_3*x**2 + 4*x_4*x**3

def generate_coefs(data, save=None):
	coefs = {}
	for i in range(len(data)):
		x, y, result = model_fit_poly(data, i, results=True)
		coefs[data[i]['probe_id'][0]] = result.best_values
	if save != None:
		save_coefs(coefs, save)
	return coefs

def data_from_coefs(coefs, space):
	if isinstance(space, int):
		space = np.linspace(0,1,space)
		space_given = False
	else:
		space_given = True
	fitted_line = np.zeros(len(space))
	for i in range(len(space)):
		fitted_line[i] = dict_poly(space[i],coefs)
	if space_given == False:
		return fitted_line, space
	else:
		return fitted_line

def save_coefs(coefs, file_type='pkl', path='', filename='probe_coefs'):
	if file_type == 'pkl':
		with open(path+filename+'.pkl','wb') as f:
			pickle.dump(coefs, f)
	elif file_type == 'json':
		with open(path+filename+'.json','w') as f:
			f.write(json.dumps(coefs))

def load_coefs(filetype='pkl', path='', filename='probe_coefs'):
	if filetype == 'pkl':
		with open(path+filename+'.pkl','rb') as f:
			return pickle.load(f)
	elif filetype == 'json':
		with open(path+filename+'.json','r') as f:
			return json.load(f)

#%% Read data

data,key = read_probes(100, rand=True)
block_data = pd.read_csv('probe_block_counts.csv')
probe_data = pd.read_csv('block_probe_counts.csv')
coefs = load_coefs()
test = pd.read_csv('gpl570_data_filtered.csv', nrows=20000)
test_probe = data[0]
test_probe_id = test_probe['probe_id'][0]
test_probe_coefs = coefs[test_probe_id]

#%%

fitted_line, space = data_from_coefs(test_probe_coefs, len(test_probe))
plt.plot(space,test_probe['p_value'].sort_values(ascending=True).values)
plt.plot(space,fitted_line)
