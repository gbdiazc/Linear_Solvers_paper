#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
® Gabriela Díaz g.b.diazcortes@gmail.com, 
  Luis López luisantoniolopezp@gmail.com
  04/2021
 -----------------------------------------------------------------------------
 -----------------------------------------------------------------------------
                         Multigrid fuctions and classes
 -----------------------------------------------------------------------------
 -----------------------------------------------------------------------------

"""

import numpy as np


import pandas as pd
from scipy.sparse import linalg as SLA
from numpy import linalg as LA
import sys
from scipy import array, linalg, dot
from scipy.sparse import csc_matrix, lil_matrix, diags
from scipy import sparse
import os
import time

#%%----------------------- Required classes ----------------------------------


# Extra grid class for the V_Cycle
class Grid_VC:
     def __init__(self,n_x,n_y,n_z,L):
         self.N_x = int((n_x)/2)
         self.N_y = int((n_y)/2)
         self.N_z = int((n_z)/2)
         self.n_x = n_x
         self.n_y = n_y
         self.n_z = n_z
         self.n   = self.n_x*self.n_y*self.n_z
         self.N   = self.N_x*self.N_y*self.N_z
         self.L   = L
         self.Nxs = []
         self.Ls  = []
          
# Multigrid method
class MGC:
     def __init__(self,x,S1_Its, S2_Its,Its,t,tp,rr,r_true,G,L,Nx_L,x_true,re):
         self.name    = 'MG '
         self.marker  = '>'
         self.x       = x
         self.t_unit  = t[2]
         self.t_init  = t[0]
         self.t_iter  = t[1]
         self.tp_init = tp[0]
         self.tp_iter = tp[1]
         self.Its     = Its
         self.rres    = rr
         self.r_true  = r_true 
         self.Nx      = G.Nx    
         self.x_true  = x_true       
         self.rerr    = re
         self.L       = L
         self.Nx_L    = Nx_L

         
class MGCGS:
     def __init__(self,x,Its,t,tp,rr,r_true,G,L,Nx_L,x_true=[],re=[]):
         self.name    = 'MGCG '
         self.marker  = 's'
         self.x       = x
         self.t_unit  = t[2]
         self.t_init  = t[0]
         self.t_iter  = t[1]
         self.tp_init = tp[0]
         self.tp_iter = tp[1]
         self.Its     = Its
         self.rres    = rr
         self.r_true  = r_true 
         self.Nx      = G.Nx    
         self.x_true  = x_true       
         self.rerr    = re
         self.L       = L
         self.Nx_L    = Nx_L
         

#%%----------------------- Required functions ----------------------------------

def init(a,f,x_0,n_x,n_y,n_z,L):
    A    = a.copy()
    b    = f.copy()
    u    = x_0.copy()
    G_VC = Grid_VC(n_x,n_y,n_z,L)
    return A,b,u,G_VC
         

def presmoothing(S1_it,A,b,u,G):
    Mv = A.diagonal()
    M_Inv  = 0.2*diags(1/Mv)
    M_IS  = csc_matrix(M_Inv)   
    I_h   = sparse.eye(G.n)
    I_hS  = csc_matrix(I_h)
 
    for i in range(0, S1_it):              
        S_h = I_hS - M_IS.dot(A)
        s_h = M_IS.dot(b.T)
        uv1 = S_h.dot(u)+s_h
        u   = uv1.copy()
    return  u  
import sys
def PR(G):
    Ix=lil_matrix((G.N_x, G.n_x))
    
    k=0
    s = np.array([1,1]).reshape(1,2)
    for i in range(0, G.N_x):

        Ix[i, k:k+2]=s
        k=k+2        

    Iy=lil_matrix((G.N_y, G.n_y))
    
    k=0
    for i in range(0, G.N_y):
        Iy[i, k:k+2]=s
        k=k+2  
    
    if G.n_z > 1:
        Iz=lil_matrix((G.N_z, G.n_z))
        k=0
        for i in range(0, G.N_z):
            Iz[i, k:k+2]=s
            k=k+2   
        Rt = sparse.kron(1/2*Ix, 1/2*Iy)
        Rt = sparse.kron(Rt, 1/2*Iz)
        Pr = 2*Rt.T
    else:
        Rt = sparse.kron(1/2*Ix, 1/2*Iy)
        Pr = 2*Rt.T

    P = csc_matrix(Pr)
    R = csc_matrix(Rt)
    return P,R
    
def restrict(r_h, A_h, P, R,G):
    r_2h   = np.zeros(G.N)
    A_2h   = lil_matrix((G.N,G.N))
    A_h_2h = lil_matrix((G.n,G.N))
    r_2h   = R.dot(r_h)    
    A_h_2h = A_h.dot(P)  
    A_2h   = R.dot(A_h_2h) 
    return r_2h, A_2h

def prolongation(e2h, P, G):
    eh = np.zeros(G.n)
    eh = P.dot(e2h)
    return eh

def correction(eh, u, G):
    un = np.zeros(G.n)
    un = u   + eh
    return un
  
def postsmoothing(S2_it,A,b,G,un):
    Mv     = A.diagonal()
    M_Inv  = 0.2*diags(1/Mv)
    M_IS   = csc_matrix(M_Inv)   
    I_h    = sparse.eye(G.n)
    I_hS   = csc_matrix(I_h)
    for i in range(0, S2_it):
        
        S_h = I_hS - M_IS.dot(A)
        s_h = M_IS.dot(b)
        uv2 = S_h.dot(un)+s_h   
    return uv2 

def v_cicle(S1_it,S2_it,A,b,u,G, Nx_L):
    G.L = G.L+1
    
    A,b,u,G_VC = init(A,b,u,G.n_x,G.n_y,G.n_z,G.L)   
    u = presmoothing(S1_it,A,b,u,G_VC)    
    Nx_L.append(G.N_x)
    r_h  = b-A.dot(u)
    P,R  = PR(G_VC)     
    
    r_2h, A_2h = restrict(r_h, A, P, R,G_VC)
    
    if G_VC.N_x < 20:
        e2h = SLA.spsolve(A_2h, r_2h)  
    else: 
        e_init = np.zeros(G_VC.N)
        e2h, L , Nx_L    = v_cicle(S1_it,S2_it, A_2h, r_2h, e_init,Grid_VC(G.N_x,G.N_y,G.N_z,G.L), Nx_L ) 
                    
    eh  = prolongation(e2h, P, G_VC)
    un  = correction(eh, u,G_VC)  
    uv2 = postsmoothing(S2_it,A,b,G_VC,un)   
   
    return uv2, G.L,  Nx_L   
    
# ---------------------- Multigrid function ----------------------------------
def MG(a,f,x_0,x_true,G, S1_it, S2_it, MG_it_max,tol):
    
    t_In_I = time.time()
    t_In_I_p = time.process_time()
 
    A,b,x,G_VC = init(a,f,x_0,G.Nx,G.Ny,G.Nz,0)   
    rr = 1
    MG_it = 0
    # Compute the relative residual and put it in a list
    r_h = b-A.dot(x)
    rr_vec = [] 
    e_vec = [] 
    rr = LA.norm(r_h)/LA.norm(b)
    rr_vec.append(rr) 
    Its_vec = []
    Its = 0
    Its_vec.append(Its) 
    
    # Cálculo del tiempo inicial
    #-------------------------------------------------------------------------
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p 
    
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)

    
    # -------------------------- Solución------------------------------------
    # Cálculo del tiempo de solución del sistema lineal, se inicializa el tiempo
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    Its = 0
    flops_init, flops_iter =  [0,0]
    flops = 0
    Nx_L = [G.Nx+1]
    while (rr > tol) and (MG_it<MG_it_max):

        if Its == 0:
            t_unit = time.time()
        G_VC.L = 0
        
        uv2,L, Nx_L    = v_cicle(S1_it,S2_it,A,b,x,G_VC,Nx_L)  
        Nx_L = [G.Nx+1]
        x     = uv2.copy()
        r_h   = b-A.dot(x)
        rr    = LA.norm(r_h)/LA.norm(b)
        MG_it = MG_it+1
 

        if Its == 0:
            t_unit_f = time.time()-t_unit
        rr_vec.append(rr)
        Its_vec.append(MG_it)
        re = LA.norm(x_true-x)/LA.norm(x_true)
        e_vec.append(re)
        Its +=1
    if Its == 0:
        t_unit = time.time()
        t_unit_f = time.time()-t_unit
        L = 0
        
    # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p
    # Cálculo del número de operaciones

    #-------------------------------------------------------------------------
    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It,t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    G.L = L
    r_true =  b - A.dot(x)
    MG_r = MGC(x,S1_it, S2_it,Its_vec,t_vec,t_vec_p,rr_vec,r_true,G,L,Nx_L,x_true,e_vec)

    return MG_r

def MGCG(a,b,x_0,x_true,G,MaxIter,tol,S1_it,S2_it):

    # ------------------------ Inicialización--------------------------------
    # Cálculo del tiempo de inicialización del método, se inicializa el tiempo
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    n = G.N
    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    r0 = 0*x_0.copy()
    r = b - A.dot(x)
    G_VC = Grid_VC(G.Nx,G.Ny,G.Nz,0)
    z = r0.copy()
    MG_it = 0
    while LA.norm(r-A*z)/LA.norm(r) > 0.05:  
        G_VC.L = 0
        Nx_L = [G.Nx]
        z,L, Nx_L  = v_cicle(S1_it,S2_it,A,r,z,G_VC,Nx_L)

    p = z

    # Compute the relative residual and put it in a list
    rr_vec = [] 
    e_vec = [] 
    rr = LA.norm(r)/LA.norm(b)
    rr_vec.append(rr) 

    # Make a list with the iterations
    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    
    # Cálculo del tiempo inicial

    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
            
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    # -------------------------- Solución------------------------------------
    # Cálculo del tiempo de solución del sistema lineal, se inicializa el tiempo
    #t_It = time.process_time()
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Método de solución 
   
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        z_old = z.copy()
        p_old = p.copy()
        if Its == 0:
            t_unit = time.time()
        w = A.dot(p_old)
        alpha = np.dot(z_old,r_old)/np.dot(w,p_old)
        x = x_old + alpha*p_old
        r = r_old - alpha*w
        #z,L = v_cicle(S1_it,S2_it,A,r,r0,G_VC)
        z = r0.copy()
        while LA.norm(r-A*z)/LA.norm(r) > 0.05:  
            # print("MGCG_it: ",Its)
            G_VC.L = 0
            z, L, Nx_L  = v_cicle(S1_it,S2_it,A,r,z,G_VC, Nx_L )
        beta = np.dot(z,r)/np.dot(z_old,r_old)
        p = z + beta*p_old 
        if Its == 0:
            t_unit_f = time.time()-t_unit
        rr = LA.norm(r)/LA.norm(b)

        re = LA.norm(x_true-x)/LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its) 
    if Its == 0:
        t_unit = time.time()
        t_unit_f = time.time()-t_unit
    
    # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p

    #-------------------------------------------------------------------------
    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true =  b - A.dot(x)
    
    MGCG_r = MGCGS(x,Its_vec,t_vec,t_vec_p,rr_vec,r_true,G,L,Nx_L,x_true,e_vec)
    return MGCG_r