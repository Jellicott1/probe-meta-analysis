#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:55:51 2020

@author: je10g15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('probe_205623_at.csv')

#%%

plt.plot(data.sort_values(by='p_value',ascending=False)['p_value'].values)
plt.scatter(np.arange(data.shape[0]),data.sort_values(by='p_value',ascending=False)['adj_p_value'].values, s=1, color='orange')
plt.legend(['p-value','Adj p-value'])
