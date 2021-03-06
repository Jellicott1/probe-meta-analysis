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
# import scipy.stats as stat
from scipy.optimize import newton
# from scipy.signal import savgol_filter
from lmfit import Model
import pickle, json
import time

#%% Data reading functions

def read_probe_list(probe_files, loc):
	try:
		data_list = {}
		for item in probe_files:
			data_list[item.strip('.csv')] = pd.read_csv('{}{}'.format(loc,item))
	except:
		raise TypeError('Invalid Probe ID(s).')
	return data_list

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

def read_matrix_file(series, loc:str='GEO-Conversions/'):
	path = loc+series+'_series_matrix.txt'
	with open(path) as f:
		i = 0
		for line in f:
			if '!series_matrix_table_begin' in line:
				start = i
				break
			else:
				i += 1				
	data = pd.read_csv(path, sep='\t', header=start, skipfooter=1, na_values='null', engine='python')	
	return data

#%%  Plotting functions

def plt_p_vs_adj_p(data, n=None):
	plt.figure()
	data = list(data.values())
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
	
def plt_many_probes(data, col='p_value', log=False, hl=None):
	plt.figure()
	if log == False:
		for item in data.values():
			plt.plot(np.linspace(1,0,len(item)),item.sort_values(by=col,ascending=False)[col].values)
		plt.title('Superimposed plots of {} for many probes'.format(col))
		plt.ylabel(col)
	else:
		for item in data.values():
			plt.plot(np.linspace(1,0,len(item)),np.log10(item.sort_values(by=col,ascending=False)[col].values))
		plt.title('Superimposed plots of log {} for many probes'.format(col))
		plt.ylabel('log {}'.format(col))
	plt.xlabel('Sample Ranking')
	if ~isinstance(hl, type(None)):
		plt.plot(np.linspace(1,0,len(hl)),hl[col].sort_values(ascending=False).values, linewidth=8)
	plt.show()
	
def plt_poly_vs_weib_fit(data, n, weight='tail'):
	fig, ax = plt.subplots(1,n)
	fig2, ax2 = plt.subplots(1,2)
	ax2[0].title.set_text('Exp-Weibull')
	ax2[1].title.set_text('Poly')
	fig.tight_layout(pad=1.0)
	probe_ids = list(data.keys())
	data = list(data.values())
	for i in range(n):
		leg = ['Probe {}'.format(probe_ids[i])]
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
		ax[i].set_xlabel('Sample Ranking')
		ax[i].set_ylabel('P-Value')
		
		if weight == 'symmetric':
			x, y, test = model_fit_poly(data, i, weight=weights)
		else:
			x, y, test = model_fit_poly(data, i)
		ax[i].plot(x,y)
		leg.append('\npoly {}\ncorr = {:.4f}\nerror = {:.2f}'.format(i, np.corrcoef(y,test)[0,1], sum(abs(y-x))))
		ax[i].legend(leg)
		ax2[1].plot(x,y)
	
def plt_fitted_curves(coefs_dict, keys, hl=None, resolution=1000):
	if isinstance(keys, int):
		keys = rnd.sample(list(coefs),k=keys)
	for item in keys:
		fitted_line, space = data_from_coefs(coefs_dict[item], resolution)
		plt.plot(space, fitted_line)
	if ~isinstance(hl, type(None)):
		fitted_line, space = data_from_coefs(coefs_dict[hl], resolution)
		plt.plot(space,fitted_line, linewidth=8, color='black')
	plt.ylim(0,1)
	plt.xlabel('Rank Within Probe')
	plt.ylabel('P-Value')
	
def plt_grad_transf(trans_data_list, space_list, data):
	for i in range(len(trans_data_list)):
		plt.plot(space_list[i],trans_data_list[i])
		plt.plot(space_list[i],data[i]['p_value'].sort_values(ascending=True).values)
	plt.xlabel('Rank Within Probe')
	plt.ylabel('P-Value/Transformed P-Value')

def plt_probe_vs_mean(probe_id, coefs, resolution=1000, probe_points=[], mean_points=[], float_points=[]):
	probe, space = data_from_coefs(coefs[probe_id], resolution)
	mean = data_from_coefs(coefs['mean'], space)
	plt.plot(space, probe)
	plt.plot(space, mean)
	for x in probe_points:
		plt.scatter(x, dict_poly(x, coefs[probe_id]), s=50)
	for x in mean_points:
		plt.scatter(x, dict_poly(x, coefs['mean']), s=50)
	for item in float_points:
		plt.scatter(item[0],item[1], s=50)
	plt.legend([probe_id, 'mean'])

def plt_trans_probe(probe_id, data, coefs, resolution=1000):
	mean_data = data_from_coefs(coefs['mean'],resolution)
	trans_data = transform_probe('47571_at', data, coefs)
	plt.plot(mean_data[1],mean_data[0])
	plt.plot(np.linspace(0,1,len(data['47571_at'])),data['47571_at']['p_value'].sort_values())
	plt.plot(trans_data[1], trans_data[0])
	plt.xlabel('Sample Ranking')
	plt.ylabel('P-Value')
	plt.legend(['Mean Probe', 'Raw Data', 'Transformed Data'])
	
def plt_trans_data(data, probes, mean=None):
	if isinstance(probes, int):
		probes = rnd.sample(data.keys(),k=probes)
	for item in probes:
		plt.plot(np.linspace(0,1,len(data[item])),data[item])
	if mean != None:
		plt.plot(mean[1], mean[0], linewidth=4, color='black')
	plt.xlabel('Sample Ranking')
	plt.ylabel('Transformed P-Value')
			

#%% Fitting functions
def poly(x, x_2, x_3, x_4):
	return x_2*x**2 + x_3*x**3 + x_4*x**4

def dict_poly(x, coefs):
	return poly(x, coefs['x_2'], coefs['x_3'], coefs['x_4'])

def expweib(x,k,lamda,alpha):
      return alpha*(k/lamda)*((x/lamda)**(k-1))*((1-np.exp(-(x/lamda)*k))**(alpha-1))*np.exp(-(x/lamda)*k)

def expweib_cdf(x,k,l,a):
	return (1-np.exp(-(x/l)**k))**a

def model_fit_poly(data, probe_id, weight='default', results=False):
	raw = data[probe_id]['p_value'].sort_values(ascending=True).values
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

def model_fit_weib(data, probe_id, weight='default'):
	raw = data[probe_id]['p_value'].sort_values(ascending=True).values
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

def poly_normal(x, probe_id, coef_dict):
	return -1/(poly_grad(x, probe_id, coef_dict))

def generate_coefs(data, save=None):
	coefs = {}
	for item in data.keys():
		x, y, result = model_fit_poly(data, item, results=True)
		coefs[item] = result.best_values
	if save != None:
		save_dict(coefs, save)
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

def save_dict(coefs, file_type='pkl', path='', filename='probe_coefs'):
	if file_type == 'pkl':
		with open(path+filename+'.pkl','wb') as f:
			pickle.dump(coefs, f)
	elif file_type == 'json':
		with open(path+filename+'.json','w') as f:
			f.write(json.dumps(coefs))

def load_dict(filetype='pkl', path='', filename='probe_coefs'):
	if filetype == 'pkl':
		with open(path+filename+'.pkl','rb') as f:
			return pickle.load(f)
	elif filetype == 'json':
		with open(path+filename+'.json','r') as f:
			return json.load(f)

#%% Transformation Functions
def gen_mean_coefs(coefs, save_to_coefs=False):
	mean_coefs = {'x_2':0,'x_3':0,'x_4':0}
	for item in coefs:
		for ind in ['x_2','x_3','x_4']:
			mean_coefs[ind] += coefs[item][ind]
	for ind in ['x_2','x_3','x_4']:
		mean_coefs[ind] = mean_coefs[ind]/len(coefs)
	if save_to_coefs == True:
		coefs['mean'] = mean_coefs
		return coefs
	else:
		return mean_coefs
	
def grad_transform(data, coefs, probe_id):
	fitted_line, space = data_from_coefs(coefs[probe_id], len(data[probe_id]))
	trans_data = np.zeros(len(data[probe_id]))
	for j in range(len(trans_data)):
		trans_data[j] = poly_grad(space[j], probe_id, coefs)*data[probe_id]['p_value'][j]
		trans_data.sort()
	return trans_data, space

def grad_transform_n(n, data, coefs):
	trans_data_list = []
	space_list =[]
	keys = data = list(data.keys())
	for i in range(n):
		trans_data, space = grad_transform(data, coefs, keys[i])
		trans_data_list.append(trans_data)
		space_list.append(space)
	return trans_data_list, space_list

def intersection_func(x, x_4, x_3, x_2, grad, c):
	return x_4*x**4 + x_3*x**3 + x_2*x**2 - grad*x + c

def intersection_func_dy_dx(x, x_4, x_3, x_2, grad, c):
	return 4*x_4*x**3 + 3*x_3*x**2 + 2*x_2*x - grad

def intersection_func_d2y_dx2(x, x_4, x_3, x_2, grad, c):
	return 12*x_4*x**2 + 6*x_3*x + 2*x_2

def call_newton_raphson(probe_id, coefs, point):
	grad = poly_normal(point[0], probe_id, coefs)
	args = [coefs['mean']['x_4'], coefs['mean']['x_3'], coefs['mean']['x_2'], grad, -point[1]+grad*point[0]]
	ans = newton(intersection_func, point[0], intersection_func_dy_dx, args, fprime2=intersection_func_d2y_dx2)
	return ans

def transform_probe(probe_id, data, coefs):
	trans_data = [0]
	data = data[probe_id]['p_value'].sort_values()
	error = []
# 	x_test = []
	space = np.linspace(0,1,len(data)+2)
	for i in range(len(data)):
		try:
			x = call_newton_raphson(probe_id, coefs, [space[i+1], dict_poly(space[i+1], coefs[probe_id])])
# 			x_test.append(x)
			if dict_poly(x,coefs['mean']) > 1:
				trans_data.append(1)
			else:
				trans_data.append(dict_poly(x,coefs['mean']))
		except RuntimeError:
			error.append([i,data[i]])
			trans_data.append(np.NaN)
	if max(trans_data) > 1:
		print('Probability greater than 1 returned')
	trans_data.append(1)
	if len(error) >= 1:
		return trans_data, error#, x_test
	else:
		return trans_data, space#, x_test

def transform_all(data, coefs, timeit=False):
	if timeit == True:
		start = time.perf_counter()
	trans_data = {}
	error_data = {}
	for probe in data.keys():
		temp_data = transform_probe(probe, data, coefs)
		trans_data[probe] = temp_data[0]
		if len(temp_data[1]) > 0:
			error_data[probe] = temp_data[1]
	if timeit == True:
		time_taken = time.perf_counter() - start
		return trans_data, time_taken, error_data
	return trans_data, error_data

def transform_sample(data, coefs):
	probe_ids = data['ID_REF']
	series_list = list(data.columns[1:])
	for series in series_list:
		for probe in probe_ids:
			x = call_newton_raphson(probe, coefs, [data[series][probe]
			
		
#%% Read data

if __name__ == '__main__':
	data = read_probes_all()#read_probes(1000, rand=True)
	block_data = pd.read_csv('probe_block_counts.csv')
	probe_data = pd.read_csv('block_probe_counts.csv')
	coefs = load_dict(filename='probe_coefs+mean')
	trans_data = load_dict(filename='transformed_probe_data', path='Transformed-Probe-Data/')
# 	trans_data_list, space_list = grad_transform_n(100, data, coefs)

#%%
# plt_trans_probe('47571_at', data, coefs)
	