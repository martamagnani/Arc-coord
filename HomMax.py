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

def multmat(B1,B2,B3,B4,C1,C2,C3,C4): #given blocks of two matrices,retunrs blocks of the matrix obtained by multiplying them
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


def mappa(A1,B1,C1,D1,E1,A2,B2,C2,D2,E2):  #Finds blocks of matrix in Sp4 sending A1,...E1 to A2,...E2
    X1=la.inv(A1-C1)-la.inv(A1-B1)
    Y1=la.inv(A1-D1)-la.inv(A1-B1)
    P1,Q1=diagonalizing_mat(la.inv(sqrtmatrix(X1))@Y1@la.inv(sqrtmatrix(X1)))
    X2=la.inv(A2-C2)-la.inv(A2-B2)
    Y2=la.inv(A2-D2)-la.inv(A2-B2)
    P2,Q2=diagonalizing_mat(la.inv(sqrtmatrix(X2))@Y2@la.inv(sqrtmatrix(X2)))
    K1,L1=Stand5uple(A1,B1,C1,D1,E1)
    K2,L2=Stand5uple(A2,B2,C2,D2,E2)
    if (K1[0][1] >0 and K2[0][1]>0) or (K1[0][1] <0 and K2[0][1]<0):   #when you will do it with your 5-uple both M1/M2 of Stand5uple will be diagonal
        R1=P1@la.inv(sqrtmatrix(X1))@la.inv(A1-B1)#but since not exactly diagonal check >/<0 to be more precise in the choise of P/Q to go up
        R2=P1@la.inv(sqrtmatrix(X1))@(Id-la.inv(A1-B1)@A1)
        R3=-P1@sqrtmatrix(X1)
        R4=P1@sqrtmatrix(X1)@A1
        H1=A2@sqrtmatrix(X2)@P2.T
        H2=(A2@la.inv(A2-B2)-Id)@la.inv(sqrtmatrix(X2))@P2.T
        H3=sqrtmatrix(X2)@P2.T
        H4=la.inv(A2-B2)@la.inv(sqrtmatrix(X2))@P2.T
        return multmat(H1,H2,H3,H4,R1,R2,R3,R4)      #note first the H matrix
    if (K1[0][1] >0 and K2[0][1]<0) or (K1[0][1] <0 and K2[0][1]>0):
        R1=P1@la.inv(sqrtmatrix(X1))@la.inv(A1-B1)
        R2=P1@la.inv(sqrtmatrix(X1))@(Id-la.inv(A1-B1)@A1)
        R3=-P1@sqrtmatrix(X1)
        R4=P1@sqrtmatrix(X1)@A1
        H1=A2@sqrtmatrix(X2)@Q2.T
        H2=(A2@la.inv(A2-B2)-Id)@la.inv(sqrtmatrix(X2))@Q2.T
        H3=sqrtmatrix(X2)@Q2.T
        H4=la.inv(A2-B2)@la.inv(sqrtmatrix(X2))@Q2.T
        return multmat(H1,H2,H3,H4,R1,R2,R3,R4)       #note first the H matrix
    
def prova(A1,B1,C1,D1,E1,A2,B2,C2,D2,E2):  #finds blocks of four matrices obtained by calculations IPad
    X1=la.inv(A1-C1)-la.inv(A1-B1)
    Y1=la.inv(A1-D1)-la.inv(A1-B1)
    P1,Q1=diagonalizing_mat(la.inv(sqrtmatrix(X1))@Y1@la.inv(sqrtmatrix(X1)))
    X2=la.inv(A2-C2)-la.inv(A2-B2)
    Y2=la.inv(A2-D2)-la.inv(A2-B2)
    P2,Q2=diagonalizing_mat(la.inv(sqrtmatrix(X2))@Y2@la.inv(sqrtmatrix(X2)))
    R1=P1@la.inv(sqrtmatrix(X1))@la.inv(A1-B1)#but since not exactly diagonal check >/<0 to be more precise in the choise of P/Q to go up
    R2=P1@la.inv(sqrtmatrix(X1))@(Id-la.inv(A1-B1)@A1)
    R3=-P1@sqrtmatrix(X1)
    R4=P1@sqrtmatrix(X1)@A1  
    S1=Q1@la.inv(sqrtmatrix(X1))@la.inv(A1-B1)#but since not exactly diagonal check >/<0 to be more precise in the choise of P/Q to go up
    S2=Q1@la.inv(sqrtmatrix(X1))@(Id-la.inv(A1-B1)@A1)
    S3=-Q1@sqrtmatrix(X1)
    S4=Q1@sqrtmatrix(X1)@A1  
    H1=A2@sqrtmatrix(X2)@P2.T
    H2=(A2@la.inv(A2-B2)-Id)@la.inv(sqrtmatrix(X2))@P2.T
    H3=sqrtmatrix(X2)@P2.T
    H4=la.inv(A2-B2)@la.inv(sqrtmatrix(X2))@P2.T
    G1=A2@sqrtmatrix(X2)@Q2.T
    G2=(A2@la.inv(A2-B2)-Id)@la.inv(sqrtmatrix(X2))@Q2.T
    G3=sqrtmatrix(X2)@Q2.T
    G4=la.inv(A2-B2)@la.inv(sqrtmatrix(X2))@Q2.T
    h1=multmat(H1,H2,H3,H4,R1,R2,R3,R4)  #note first the H matrix
    h2=multmat(H1,H2,H3,H4,S1,S2,S3,S4) 
    h3=multmat(G1,G2,G3,G4,R1,R2,R3,R4) 
    h4=multmat(G1,G2,G3,G4,S1,S2,S3,S4)
    return h1,h2,h3,h4
    
    
    
def conjugating_mat(M,N): #takes M,N symm of same ev returns P,Q in SO(2) and O(2)\SO(2) such that P@M@P.T=Q@M@Q.T=N
    P1,Q1=diagonalizing_mat(M)
    P2,Q2=diagonalizing_mat(N)
    return P2.T@P1,P2.T@Q1

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

c1=10
c2=3

d1=8
d2=2

S1=special_ortho_group.rvs(dim=2)    #generates random matrix in SO(2)
S2=special_ortho_group.rvs(dim=2)

T1=special_ortho_group.rvs(dim=2)
T2=special_ortho_group.rvs(dim=2)



#Esagono verde al centro  (infty,-G,-E,0,D,E,Z3,Id,F,Z4,G,H)

Id=np.identity(2)  
F=np.array([[math.exp(c1),0],[0,math.exp(c2)]])
E=ProblemaL1(b1,b2,c1,c2,S1,Id,F)
G=ProblemaL2(c1,c2,d1,d2,S2,Id,F)
D=E@E
H=G@(la.inv(F))@G
Z3,Z4=OrthTube(D,Id,F,H)


#Esagono  giallo in alto (infty,-L,-G,0,F,G,Z5,H,I,Z6,L,M)

I=ProblemaL2(d1,d2,c1,c2,T1,G,H)    
L=ProblemaL2(c1,c2,b1,b2,T2,H,I)    
M=L@(la.inv(I))@L
Z5,Z6=OrthTube(F,H,I,M)

#Esagono giallo in alto ribaltato (infty,-L_r,-G,0,F,G,Z5_r,H,I_r,Z6_r,L_r,M_r)
T1_r=ribalta(T1)
T2_r=ribalta(T2)
I_r=ProblemaL2(d1,d2,c1,c2,T1_r,G,H)    
L_r=ProblemaL2(c1,c2,b1,b2,T2_r,H,I_r)    
M_r=L_r@(la.inv(I_r))@L_r
Z5_r,Z6_r=OrthTube(F,H,I_r,M_r)

#Esagono giallo in basso (infty,-E,-B,0,A,B,Z1,C,D,Z2,E,Id)

C=ProblemaL1(c1,c2,b1,b2,T2,D,E)
B=ProblemaL1(d1,d2,c1,c2,T1,C,D)
A=B@la.inv(C)@B
Z1,Z2=OrthTube(A,C,D,Id)

#Altro esagono verde 



#First generator (one of them sends hexagon to the flipped one above)

P,Q=conjugating_mat(la.inv(sqrtmatrix(B))@C@la.inv(sqrtmatrix(B)),la.inv(sqrtmatrix(G))@H@la.inv(sqrtmatrix(G)))
A1=sqrtmatrix(G)@P@la.inv(sqrtmatrix(B))
A2=sqrtmatrix(G)@Q@la.inv(sqrtmatrix(B))
g1=np.block([[A1,np.zeros((2,2))],[np.zeros((2,2)),la.inv(A1.T)]])
g2=np.block([[A2,np.zeros((2,2))],[np.zeros((2,2)),la.inv(A2.T)]])    


#Second generator

M1,M2,M3,M4=prova(D,Id,F,Z4,H,D,Id,A,Z1,C)  #Ognuna sono i quattro blocchi di una matrice che manda (D,Id,F,Z4,H) in (D,Id,A,Z1,C) 
H1=np.block([[M1[0],M1[1]],[M1[2],M1[3]]])   #coincideranno a due a due e saranno h1 e h2 
H2=np.block([[M2[0],M2[1]],[M2[2],M2[3]]])
H3=np.block([[M3[0],M3[1]],[M3[2],M3[3]]])
H4=np.block([[M4[0],M4[1]],[M4[2],M4[3]]])
bl=conj(Id,D)                                 #blocchi di una matrice g| ghg^{-1} fissa {0},infty} dove h fissa {Id,D}
g=np.block([[bl[0],bl[1]],[bl[2],bl[3]]])     #quindi h è una tra H1,...H4
a=checkflip(g@H1@la.inv(g))                   #ora controllo quale fra H1,...,H4 sono uguali (flippano o no)
if a==checkflip(g@H2@la.inv(g)):
    h1,h2=H1,H3
else:
    h1,h2=H1,H2

#Third generator

K1,K2,K3,K4=mappa(F,H,I,Z6,M,F,H,D,Z3,Id)
k1=np.block([[K1,K2],[K3,K4]])









