#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:34:16 2018

@author: vash
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.linspace(0,20,50)
y=x*2+1+1*np.sin(4*x)+0.1*np.random.random(50)
u=np.ones(50)
df=pd.DataFrame({'x':x,'y':y})

# curve fit from scipy
from scipy import optimize
def lin_func(x,a,b):
    return a*x+b
linfit_coefs,linfit_covar=optimize.curve_fit(approx_func,x,y,[1,1])
print(linfit_coefs)
def lin_sin_func(x,a,b,c):
    return a*x+b+c*np.sin(4*x)
linsinfit_coefs,linsinfit_covar=optimize.curve_fit(lin_sin_func,x,y,[1,1,1])
print(linsinfit_coefs)

# matrix approach for linear reg
def std_linreg(X,Y):
    XTX=X.T*X
    if np.linalg.det(XTX) == 0:
        print('singular matrix')
        return
    return XTX.I*X.T*Y
mx=np.mat(np.stack((x,u),axis=1))
my=np.mat(y).T
linmat_coef=std_linreg(mx,my)
print(linmat_coef)


