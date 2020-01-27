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

#%%

data,key = read_probes(3,rand=True)

#%%
fig, ax = plt.subplots(1,3)
for i in range(len(data)):
    ax[i].plot(data[i].sort_values(by='p_value',ascending=False)['p_value'].values)
    ax[i].scatter(np.arange(data[i].shape[0]),data[i].sort_values(by='p_value',ascending=False)['adj_p_value'].values, s=1, color='orange')
fig.legend(['p-value','Adj p-value'],loc='upper center')

#%%
block_data = pd.read_csv('block_probe_counts.csv')

plt.figure()
plt.hist(block_data['Unique Probes'],bins=40)
plt.ylabel('Density')
plt.xlabel('Number of Probes')
plt.show()

block_data.describe()
stat.iqr(block_data['Unique Probes'])
stat.tstd(block_data['Unique Probes'])
