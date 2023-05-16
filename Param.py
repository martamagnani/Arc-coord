#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:25:59 2023

@author: MartaMagnani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:52:09 2021

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

def diagonalizing_mat(M): #takes M symm returns P,Q in SO(2) and O(2)\SO(2) such that P@M@P.T and Q@M@Q.T diagonal with first eigenvalue > second
    eigvals,eigvecs=la.eig(M)
    L=eigvecs      #columns are orthonormal eigenvectors of M
    if eigvals[0]<eigvals[1]:  #if first eigenvalue smaller then the second I swap columns of L (now first col rel to bigger ev)
        l=[1,0]    #per switchare non posso salvare in temp, con np è puntatore quindi si modifica modificando colonne
        L=L[:,l]  
    J=np.array([[-1,0],[0,1]])
    if la.det(L)>0:
        P=L.copy()
        Q=P@J
    if la.det(L)<0:
        Q=L.copy()
        P=Q@J
    return P.T,Q.T

   
def Stand5uple(A,B,C,D,E):    #returns M1,M2 where (infty,0,Id,diag, M1/M2) is the standard 5-uple to which A,...E are sent
    X=la.inv(A-C)-la.inv(A-B)
    Y=la.inv(A-D)-la.inv(A-B)
    Z=la.inv(A-E)-la.inv(A-B)
    P,Q=diagonalizing_mat(la.inv(sqrtmatrix(X))@Y@la.inv(sqrtmatrix(X)))
    #print(P@la.inv(sqrtmatrix(X))@Y@la.inv(sqrtmatrix(X))@P.T)  #queste diag
    #print(Q@la.inv(sqrtmatrix(X))@Y@la.inv(sqrtmatrix(X))@Q.T)
    return P@la.inv(sqrtmatrix(X))@Z@la.inv(sqrtmatrix(X))@P.T,Q@la.inv(sqrtmatrix(X))@Z@la.inv(sqrtmatrix(X))@Q.T

def multmat(B1,B2,B3,B4,C1,C2,C3,C4): #given blocks of two matrices,returns blocks of the matrix obtained by multiplying them
    return B1@C1+B2@C3,B1@C2+B2@C4,B3@C1+B4@C3,B3@C2+B4@C4

def conj(A,B): #takes map in PSP(4,R) fixing A,B and returns blocks of g wehere g| ghg^{-1} fixes 0,infty
    Id=np.identity(2) 
    M=la.inv(A-B)
    return M,Id-M@A,-Id,A  #now ghg^{-1} has the diagonal form

def checkflip(M): #takes g 4x4 matrix \in Stab(0,infty) and checks if flips (detA>0 or <0)
    d=M[0][0]*M[0][1]-M[1][0]*M[1][1]
    if d>0:
        return 1
    if d<0:
        return -1
    
def checkflipgen(A,B,C,D,X,Y): #takes g 4x4 matrix \in Stab(X,Y) and checks if conj to one flipping (detA< or >0)
     Id=np.identity(2)
     B1,B2,B3,B4=multmat(A,B,C,D,Y,Y@la.inv(Y-X)-Id,Id,la.inv(Y-X))
     A=multmat(la.inv(Y-X),Id-la.inv(Y-X)@Y,-Id,Y,B1,B2,B3,B4)[0]
     if la.det(A)>0:
            return 1
     if la.det(A)<0:
        return -1

def extractblocks(M): #takes4x4  matrix and returns the four blocks
    B1=np.array([[M[0][0],M[0][1]],[M[1][0],M[1][1]]])
    B2=np.array([[M[0][2],M[0][3]],[M[1][2],M[1][3]]])
    B3=np.array([[M[2][0],M[2][1]],[M[3][0],M[3][1]]])
    B4=np.array([[M[2][2],M[2][3]],[M[3][2],M[3][3]]])
    return B1,B2,B3,B4
  
    
def conjugating_mat(M,N): #takes M,N symm of same ev returns P,Q in SO(2) and O(2)\SO(2) such that P@M@P.T=Q@M@Q.T=N
    P1,Q1=diagonalizing_mat(M)
    P2,Q2=diagonalizing_mat(N)
    return P2.T@P1,P2.T@Q1

def findmaps(B,C,D,E,F,A):   #finds the two maps h such that h(B,E,F,D)=(B,E,A,C)
    Id=np.identity(2)    
    g=np.block([[la.inv(E-B),Id-la.inv(E-B)@E],[-Id,E]]) 
    M=sqrtmatrix((E-D)@la.inv(D-B)@(E-B))@la.inv(E-F)@(B-F)@la.inv(E-B)@sqrtmatrix((E-D)@la.inv(D-B)@(E-B))
    N=sqrtmatrix((E-C)@la.inv(C-B)@(E-B))@la.inv(E-A)@(B-A)@la.inv(E-B)@sqrtmatrix((E-C)@la.inv(C-B)@(E-B))
    P,Q=conjugating_mat(M,N)
    G1=sqrtmatrix(la.inv(E-C)@(C-B)@la.inv(E-B))@P@sqrtmatrix((E-D)@la.inv(D-B)@(E-B))
    G2=sqrtmatrix(la.inv(E-C)@(C-B)@la.inv(E-B))@Q@sqrtmatrix((E-D)@la.inv(D-B)@(E-B))
    s=(2,2)
    Z=np.zeros(s)
    h1=np.block([[G1,Z],[Z,la.inv(G1.T)]])
    h2=np.block([[G2,Z],[Z,la.inv(G2.T)]])
    return la.inv(g)@h1@g,la.inv(g)@h2@g  #the first map is the one reversing


def Sp4_Action(A,B,C,D,Z):  
    return (A@Z+B)@la.inv(C@Z+D)

def OrthTube(A,B,C,D):  #Given a max 4-uple returns endpts of unique tube orth to Y_AB and Y_CD (Z1 btw AB, Z2 btw CD)
    X=la.inv(A-C)-la.inv(A-B)
    Y=la.inv(A-D)-la.inv(A-B)
    Q=la.inv(sqrtmatrix(X))@Y@la.inv(sqrtmatrix(X))
    Z1=Sp4_Action(A@sqrtmatrix(X),-la.inv(sqrtmatrix(X))+A@la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(X),la.inv(A-B)@la.inv(sqrtmatrix(X)),-sqrtmatrix(Q))
    Z2=Sp4_Action(A@sqrtmatrix(X),-la.inv(sqrtmatrix(X))+A@la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(X),la.inv(A-B)@la.inv(sqrtmatrix(X)),sqrtmatrix(Q))
    return Z1,Z2

def ProblemaL1(c1,c2,d1,d2,S,Y,Z): #(left arrow) given parameters+last two matrices I get first matrix
    des1=[1/math.exp(c2),1/math.exp(c1)]
    C=np.diag(des1)
    X_pr=S.T@C@S
    P=diagonalizing_mat(la.inv(sqrtmatrix(Y))@Z@la.inv(sqrtmatrix(Y)))[0]
    X=sqrtmatrix(Y)@P.T@X_pr@P@sqrtmatrix(Y)
    return X

def ProblemaL2(c1,c2,d1,d2,S,X,Y): #(left arrow) given parameters+first two matrices I get last matrix
    P=diagonalizing_mat(la.inv(sqrtmatrix(Y))@X@la.inv(sqrtmatrix(Y)))[0]
    S_h=S.T@P
    des2=[math.exp(d1),math.exp(d2)]
    D=np.diag(des2)
    Z=sqrtmatrix(Y)@S_h.T@D@S_h@sqrtmatrix(Y)
    return Z

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

def Weyl_distance(A,B):    #gives Weyl chamber distance d(iA,iB) where A<B
    l1,l2=ord_eigvals(la.inv(A)@B)
    return math.log(l1),math.log(l2)

def CrossRatio(X1,X2,X3,X4):
    R=la.inv(X1-X2)@(X4-X2)@la.inv(X4-X3)@(X1-X3)
    return  R

def ribalta(S):
    J=np.array([[-1,0],[0,1]])
    return J@S@J


#Parameters for a pair of pants (three vectorial lengths and two angles per hexagon)

b1=9
b2=3

c1=7
c2=4

d1=8
d2=5

S1=special_ortho_group.rvs(dim=2)    #generates random matrix in SO(2)
S2=special_ortho_group.rvs(dim=2)

r1,r2,r3,r4=1,0,0,1

#First yellow hexagon


if r1==0:
    Id=np.identity(2)  
    C=np.array([[math.exp(c1),0],[0,math.exp(c2)]])
    A=ProblemaL1(b1,b2,c1,c2,S1,Id,C)
    D=ProblemaL2(c1,c2,d1,d2,S2,Id,C)
    Z1,Z2=OrthTube(A@A,Id,C,D@la.inv(C)@D)
if r1==1:
    Id=np.identity(2)  
    C=np.array([[math.exp(c1),0],[0,math.exp(c2)]])
    A=ProblemaL1(b1,b2,c1,c2,ribalta(S1),Id,C)
    D=ProblemaL2(c1,c2,d1,d2,ribalta(S2),Id,C)
    Z1,Z2=OrthTube(A@A,Id,C,D@la.inv(C)@D)

#First green hexagon


if r2==0:
        E=ProblemaL2(d1,d2,c1,c2,S2,D,D@la.inv(C)@D)    
        F=ProblemaL2(c1,c2,b1,b2,S1,D@la.inv(C)@D,E)    
        Z3,Z4=OrthTube(C,D@la.inv(C)@D,E,F@la.inv(E)@F)
if r2==1:
    E=ProblemaL2(d1,d2,c1,c2,ribalta(S2),D,D@la.inv(C)@D)    
    F=ProblemaL2(c1,c2,b1,b2,ribalta(S1),D@la.inv(C)@D,E)    
    Z3,Z4=OrthTube(C,D@la.inv(C)@D,E,F@la.inv(E)@F)

#Second yellow hexagon


if r3==0:
    G=ProblemaL2(b1,b2,c1,c2,S1,F,F@la.inv(E)@F)
    H=ProblemaL2(c1,c2,d1,d2,S2,F@la.inv(E)@F,G)
    Z5,Z6=OrthTube(E,F@la.inv(E)@F,G,H@la.inv(G)@H)
if r3==1:
    G=ProblemaL2(b1,b2,c1,c2,ribalta(S1),F,F@la.inv(E)@F)
    H=ProblemaL2(c1,c2,d1,d2,ribalta(S2),F@la.inv(E)@F,G)
    Z5,Z6=OrthTube(E,F@la.inv(E)@F,G,H@la.inv(G)@H)




#Second green hexagon
h1,h2=findmaps(C,Z2,Z3,D@la.inv(C)@D,E,A@A)


if r4==0:
    B1,B2,B3,B4=extractblocks(h1)
    X1=Sp4_Action(B1, B2, B3, B4, F)
    X2=Sp4_Action(B1, B2, B3, B4, -F)
    X3=Sp4_Action(B1, B2, B3, B4, -D)
    X4=Sp4_Action(B1, B2, B3, B4, D)
    Z7,Z8=OrthTube(X1,X2,X3,X4)
if r4==1:
    B1,B2,B3,B4=extractblocks(h2)
    X1=Sp4_Action(B1, B2, B3, B4, F)
    X2=Sp4_Action(B1, B2, B3, B4, -F)
    X3=Sp4_Action(B1, B2, B3, B4, -D)
    X4=Sp4_Action(B1, B2, B3, B4, D)
    Z7,Z8=OrthTube(X1,X2,X3,X4)




print(X1)
print(X2)
print(X3)
print(X4)


