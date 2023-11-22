#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:20:30 2023

@author: MartaMagnani
"""

import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm



def CrossRatio(X1,X2,X3,X4):
    R=la.inv(X1-X2)@(X4-X2)@la.inv(X4-X3)@(X1-X3)
    return  R

def switch(a,b): 
    tmp = a
    a = b
    b = tmp
    return a,b

def ord_eigvals(M):      #given a matrix M returns e1,e2 eigenvalues s.t. e1>e2
    eigvals,eigvecs=la.eig(M)
    eigvals=eigvals.real
    e1=eigvals[0]
    e2=eigvals[1]
    if e1<e2:
        e1,e2=switch(e1,e2) 
    return e1,e2

Id=np.identity(2)  

b1=40  #vector b 
b2=0.01
if b1<b2 or b2<0:
    print('b not in Weyl chamber')
    
d1=35  #vector d
d2=0.01
if d1<d2 or d2<0:
    print('d not in Weyl chamber')
    
alpha1=0 #first angle
alpha2=0 #second angle

S1= np.array([[math.cos(alpha1/2),-math.sin(alpha1/2)],[math.sin(alpha1/2),math.cos(alpha1/2)]])
S2= np.array([[math.cos(alpha2/2),-math.sin(alpha2/2)],[math.sin(alpha2/2),math.cos(alpha2/2)]])

A=S1.T@(np.array([[1/math.exp(b2),0],[0,1/math.exp(b1)]]))@S1
X=A@A


# Placing the two plots in the plane
plot1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
plot2 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)

n=50     #square where the Weyl chamber lives
x=[0,n] 
y=[0,n]
plot1.plot(x,y,color='gray')        #plot diagonal line of Weyl chamber
plot2.plot(x,y,color='gray')



k=10   #number of lines
color = iter(cm.rainbow(np.linspace(0, 1, k+1)))  #k number of colours
for l in range(k+1):
    m=l/k
    j = next(color)
    for c1 in range(1,n):
        c2=m*c1
        plot1.scatter(c1,c2, color=j, marker='o')  #plot on the first plot points on lines
        C=np.array([[math.exp(c1),0],[0,math.exp(c2)]])
        M=np.array([[0,math.sqrt(math.exp(c1))],[-math.sqrt(math.exp(c2)),0]])
        D=M@S2@(np.array([[math.exp(d1),0],[0,math.exp(d2)]]))@S2.T@M.T
        Y=D@la.inv(C)@D
        R=CrossRatio(X,Id,C,Y)
        (r1,r2)=ord_eigvals(R)
        (l1,l2)=(math.log(r1),math.log(r2))
        plot2.scatter(l1,l2, color=j, marker='o')
        #plot2.scatter(c1*c1*c1,c2, color=j, marker='o')

plt.tight_layout()        
plt.show()
    
            

        







