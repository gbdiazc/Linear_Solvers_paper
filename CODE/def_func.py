# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:24:55 2020

@author: Dell
"""
from scipy.sparse import *
from scipy import *
import numpy as np
from scipy.sparse import linalg as SLA
def def_vect(dv,G):
    if dv == 'SD':
        Z = lil_matrix((G.Nx*G.Ny,4))
        nz = int(G.N/4)
        for  i in [0,1,2,3]:
            Z[i*nz:(i+1)*nz,i] =1
    elif dv == 'Eigs' :   
         Mat_MA =  MA_f(A,M,M2)
         Mat_MA = csc_matrix(Mat_MA)
         eival  ,Z  = SLA.eigsh(Mat_MA ,G.N-1,return_eigenvectors=True)  
         Z = lil_matrix( Z[:][:,900:])
    return Z

def Z_sub(Nx,Z_vec):
    # Compute subdomain deflation matrix,containing Z_vec
    # deflation vectors, for a square matrix of size Nx*Nx
    N = Nx*Nx
    Z_el = int(N/Z_vec)
    
    row = [x for x in range(0,N)]
    col = []
    for i in range(0,Z_vec):
        col.append(i*np.ones(Z_el))
    col = [y for x in col for y in x]
    data = np.ones(N)
    Z = csr_matrix( (data,(row,col)), shape=(N,Z_vec) )
    return Z
def Deflation_P(B,Z,r,T=False):
    if T == True:
        s = r - Z.dot( B.transpose().dot(r))
    else:
        s = r - B.dot(Z.transpose().dot(r))
    return s
def Correction_Q(Z,EI,x):
    q = Z.dot(EI.dot(Z.transpose().dot( x)))
    return q

    
    