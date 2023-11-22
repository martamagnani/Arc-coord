#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:26:50 2023

@author: MartaMagnani
"""


import numpy as np
from numpy import linalg as la
import math
from scipy.stats import special_ortho_group


def sqrtmatrix(M):  #Returns square root of a positive definite matrix
    eigvals,eigvecs=la.eig(M)
    eigvals=eigvals.real
    S=eigvecs              #columns are orthonormal eigenvectors of M
    s0=math.sqrt(eigvals[0])
    s1=math.sqrt(eigvals[1])
    D=np.array([[s0,0],[0,s1]]) 
    return S@D@S.T

def diagonalizing_mat(M): #takes M symm returns P,Q in SO(2) and O(2)\SO(2) resp. such that P@M@P.T and Q@M@Q.T diagonal with first eigenvalue > second
    eigvals,eigvecs=la.eig(M)
    L=eigvecs      #columns are orthonormal eigenvectors of M
    if eigvals[0]<eigvals[1]:  #if first eigenvalue smaller then the second I swap columns of L (now first col rel to bigger ev)
        l=[1,0]  
        L=L[:,l]  
    J=np.array([[-1,0],[0,1]])
    if la.det(L)>0:
        P=L.copy()
        Q=P@J
    if la.det(L)<0:
        Q=L.copy()
        P=Q@J
    return P.T,Q.T

def ProblemaL1(c1,c2,d1,d2,S,Y,Z): #given parameters+last two matrices returns first matrix
    des1=[1/math.exp(c2),1/math.exp(c1)]
    C=np.diag(des1)
    X_pr=S.T@C@S
    P=diagonalizing_mat(la.inv(sqrtmatrix(Y))@Z@la.inv(sqrtmatrix(Y)))[0]
    X=sqrtmatrix(Y)@P.T@X_pr@P@sqrtmatrix(Y)
    return X

def ProblemaL2(c1,c2,d1,d2,S,X,Y): #given parameters+first two matrices returns last matrix
    P=diagonalizing_mat(la.inv(sqrtmatrix(Y))@X@la.inv(sqrtmatrix(Y)))[0]
    S_h=S.T@P
    des2=[math.exp(d1),math.exp(d2)]
    D=np.diag(des2)
    Z=sqrtmatrix(Y)@S_h.T@D@S_h@sqrtmatrix(Y)
    return Z

def Sp4_Action(A,B,C,D,Z):  
    return (A@Z+B)@la.inv(C@Z+D)

def OrthTube(A,B,C,D):  #Given a max 4-uple returns endpts of unique tube orth to Y_AB and Y_CD (Z1 between A&B, Z2 between C&D)
    X=la.inv(A-C)-la.inv(A-B)
    Y=la.inv(A-D)-la.inv(A-B)
    Q=la.inv(sqrtmatrix(X))@Y@la.inv(sqrtmatrix(X))
    Z1=Sp4_Action(A@sqrtmatrix(X),-la.inv(sqrtmatrix(X))+A@la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(X),la.inv(A-B)@la.inv(sqrtmatrix(X)),-sqrtmatrix(Q))
    Z2=Sp4_Action(A@sqrtmatrix(X),-la.inv(sqrtmatrix(X))+A@la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(X),la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(Q))
    return Z1,Z2


def Stand4uple(A,B,C,D):    #returns blocks of a matrix g in Sp(4,R) such that g(A,B,C,D)=(0,Id,Y,infty) with Y diagonal
    X=la.inv(D-B)-la.inv(D-A)
    W=la.inv(D-C)-la.inv(D-A)
    P=diagonalizing_mat(la.inv(sqrtmatrix(X))@W@la.inv(sqrtmatrix(X)))[0]
    Id=np.identity(2)  
    B1=P@la.inv(sqrtmatrix(X))@la.inv(D-A)
    B2=P@la.inv(sqrtmatrix(X))@(Id-(la.inv(D-A)@D))
    B3=-P@sqrtmatrix(X)
    B4=P@sqrtmatrix(X)@D
    return B1,B2,B3,B4

def multmat(B1,B2,B3,B4,C1,C2,C3,C4): #given blocks of two matrices,retunrs blocks of the matrix obtained by multiplying them
    return B1@C1+B2@C3,B1@C2+B2@C4,B3@C1+B4@C3,B3@C2+B4@C4

def mult3mat(B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,D4): #given blocks of three matrices,retunrs blocks of the matrix obtained by multiplying them
    A1,A2,A3,A4=multmat(C1,C2,C3,C4,D1,D2,D3,D4)
    return multmat(B1, B2, B3, B4, A1, A2, A3, A4)

def switch(a,b): 
    tmp = a
    a = b
    b = tmp
    return a,b

def ord_eigvals(M):      #given a matrix M returns e1,e2 eigenvalues s.t. e1>e2
    e1,e2=la.eigvals(M)
    if e1<e2:
        e1,e2=switch(e1,e2) 
    return e1,e2

def Weyl_distance(A,B):    #gives Weyl chamber distance d(iA,iB) where (0,A,B,infty) maximal
    l1,l2=ord_eigvals(la.inv(A)@B)
    return math.log(l1),math.log(l2)


def findangle(X,Y,Z): #takes (0,X,Y,Z,infty) max returns angle parameter in matrix form
     P=diagonalizing_mat(la.inv(sqrtmatrix(Y))@Z@la.inv(sqrtmatrix(Y)))[0]
     return diagonalizing_mat(P@la.inv(sqrtmatrix(Y))@X@la.inv(sqrtmatrix(Y))@P.T)[0]
 
def checkflip(g1): #takes g1 (eg block of the matrix g in Stab(0,infty) ) and checks if it is flipping H^2 component (detg1>0 or <0)
    d=g1[0][0]*g1[0][1]-g1[1][0]*g1[1][1]
    if d>0:
        return 1
    if d<0:
        return -1    
 
def checkflipgen(A,B,C,D,X,Y): #takes 4 blocks of h in Stab(X,Y) and checks if it is flipping H^2 component (conj to one in Stab(0,infty) flipping)
     Id=np.identity(2)
     B1,B2,B3,B4=multmat(A,B,C,D,Y,Y@la.inv(Y-X)-Id,Id,la.inv(Y-X))
     A=multmat(la.inv(Y-X),Id-la.inv(Y-X)@Y,-Id,Y,B1,B2,B3,B4)[0]
     if la.det(A)>0:
            return 1
     if la.det(A)<0:
        return -1    

#Length parameters b,c,d:
    
b1=9
b2=3

c1=7
c2=4

d1=8
d2=5


#Angle parameters alpha1,alpha2: 
    
alpha1=math.pi/4 
alpha2=math.pi/5 

#Matrix expression of angle parameters:
    
S1= np.array([[math.cos(alpha1/2),-math.sin(alpha1/2)],[math.sin(alpha1/2),math.cos(alpha1/2)]])
S2= np.array([[math.cos(alpha2/2),-math.sin(alpha2/2)],[math.sin(alpha2/2),math.cos(alpha2/2)]])

#Parameters in K in the case where the  hexagon is non-generic: 1=R_st, -1=R_ex

R1=1
R2=-1
R3=1


#Construction of hexagon with arc coordinates (b,c,dalpha1,alpha2):
    
Id=np.identity(2)  
C=np.array([[math.exp(c1),0],[0,math.exp(c2)]])  
A=ProblemaL1(b1,b2,c1,c2,S1,Id,C)
D=ProblemaL2(c1,c2,d1,d2,S2,Id,C)
Z1,Z2=OrthTube(A@A,Id,C,D@la.inv(C)@D)


r=np.array([[-1,0],[0,1]])
s=(2,2)
Z=np.zeros(s)   #zero matrix


G1,G2,G3,G4=Stand4uple(-D,Z,C,D)        #blocks of G| G(-D,Z,C,D)=(0,Id,Y,infty) Y diagonal
M1,M2,M3,M4=Stand4uple(-A, Z, A@A, A)   #blocks of M| M(-A, Z, A@A, A)=(0,Id,Y,infty) Y diagonal
N1,N2,N3,N4=Stand4uple(Z1, Id, C, Z2)   #blocks of N| N(Z1, Id, C, Z2)=(0,Id,Y,infty) Y diagonal

#Blocks of the reflection fixing -D,D and switching 0,infty and C,DC^{1}D
if R1==1:
    B1,B2,B3,B4=mult3mat(G4.T, -G2.T, -G3.T, G1.T, -Id, Z, Z, Id, G1, G2, G3, G4)
if R1==-1:
    B1,B2,B3,B4=mult3mat(G4.T, -G2.T, -G3.T, G1.T, -r, Z, Z, r, G1, G2, G3, G4)
  
#Blocks of reflection fixing -A,A and switching 0,infty and A^{2},Id    
if R2==1:
    C1,C2,C3,C4=mult3mat(M4.T, -M2.T, -M3.T, M1.T, -Id, Z, Z, Id, M1, M2, M3, M4)
if R2==-1:
    C1,C2,C3,C4=mult3mat(M4.T, -M2.T, -M3.T, M1.T, -r, Z, Z, r, M1, M2, M3, M4)   
 
#Blocks of reflection fixing Z1,Z2 and switching A^{2},Id and C,DC^{1}D
if R3==1:
    D1,D2,D3,D4=mult3mat(N4.T, -N2.T, -N3.T, N1.T, -Id, Z, Z, Id, N1, N2, N3, N4)
if R1==-1:
    D1,D2,D3,D4=mult3mat(N4.T, -N2.T, -N3.T, N1.T, -r, Z, Z, r, N1, N2, N3, N4)    
    
#Blocks of the first generator    
g1,g2,g3,g4=multmat(B1, B2, B3, B4, C1, C2, C3, C4) 
#g=np.block([[g1,g2],[g3,g4]])

#Blocks of the second generator
h1,h2,h3,h4=multmat(C1, C2, C3, C4, D1, D2, D3, D4)    
#h=np.block([[h1,h2],[h3,h4]])    
    
    