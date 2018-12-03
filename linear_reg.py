#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:34:16 2018

@author: vash
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

n=200
x=np.linspace(0,20,n)
y=x*2+1+1*np.sin(4*x)+0.2*np.random.random(n)
u=np.ones(n)
df=pd.DataFrame({'x':x,'y':y})

# curve fit from scipy
from scipy import optimize
def lin_func(x,a,b):
    return a*x+b
linfit_coefs,linfit_covar=optimize.curve_fit(lin_func,x,y,[1,1])
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

# locally weighted linear regression
# local linear regression on x0 with lambda=k
def loc_linreg(X,Y,x0,k):
    def gaussian_kernel(v,k):
        return np.exp(-np.linalg.norm(v,2)/(2*k))/k
    N=len(Y)
    W=np.mat(np.zeros((N,N)))
    for i in range(N):
        W[i,i]=gaussian_kernel(X[i]-x0,k)
    return (X.T*W*X).I*X.T*W*Y
pred_y=[]
for xi in mx:
    w=loc_linreg(mx,my,xi,0.1)
    #w=loc_linreg(mx,my,xi,0.5)
    #w=loc_linreg(mx,my,xi,0.05)
    pred_y.append(xi*w)
py=[float(t) for t in pred_y]
plt.figure(figsize=(16,12))
plt.scatter(x,y,c='r',s=4)
plt.plot(x,py,c='b')
plt.show()