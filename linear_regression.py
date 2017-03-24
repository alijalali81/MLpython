# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:13:53 2017

@author: ajalali
"""

import numpy as np
import matplotlib.pyplot as plt

#x=np.random.randn(100)
#y=5*(x**2)-2*x+0.1



class LinearRegression:
    
    trained_model=[0,0]
    alpha=0.1
    
    def __init__(self,inp,d_out,m_order):
        self.input=inp
        self.desired_output=d_out
        self.model_order=m_order
        
    def lms_single(self):
        X=np.zeros([np.size(self.input),self.model_order+1])
        x=self.input
        y=self.desired_output
        for i in range(0,self.model_order+1):
            X[:,i]=x**i
            
        xx=np.matmul(np.transpose(X),X)
        xx_inv=np.linalg.inv(xx)
        xxx=np.matmul(xx_inv,np.transpose(X))
        theta=np.matmul(xxx,y)
        self.theta=theta
        
    def lms_multi(self):
        X=self.input
        y=self.desired_output            
        xx=np.matmul(np.transpose(X),X)
        xx_inv=np.linalg.inv(xx)
        xxx=np.matmul(xx_inv,np.transpose(X))
        theta=np.matmul(xxx,y)
        self.theta=theta
        
    def stochastic_lms(self):
        theta=np.random.randn(self.model_order+1)
        error=np.zeros(np.size(self.input))
        alpha=self.alpha
        X=np.zeros([np.size(self.input),self.model_order+1])
        x=self.input
        y=self.desired_output        
        t_i=[]
        for i in range(0,np.size(self.input)):
            X=np.zeros(self.model_order+1)
            xx=x[i]
            for j in range(0,self.model_order+1):
                X[j]=xx**j
                
            h_theta=np.dot(theta,X)
            er=y[i]-h_theta
            theta=np.add(theta,alpha*er*X)
            error[i]=er
            t_i.insert(len(t_i),theta)
            
        self.theta=theta
        self.theta_history=t_i
        self.error=error
        
#plt.plot(er1)          
        
    