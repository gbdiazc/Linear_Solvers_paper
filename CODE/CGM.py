#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:21:30 2019

@author: gaby
"""
import numpy as np
import time
import numpy.linalg as LA
from scipy.sparse import linalg as SLA
import numpy.random
from scipy.sparse import csc_matrix, lil_matrix
from scipy import sparse
import sys
from def_func import *
from scipy.sparse import spdiags



# Conjugate gradient method 
class CGS:
     def __init__(self,x,Its,t,tp,rr,r_true,G,x_true=[],re=[]):
         self.name    = 'CG'
         self.marker  = 'd'
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
             
# Conjugate gradient method preconditioned  
class PCGS:
     def __init__(self,x,Its,t,tp,rr,r_true,G,pre,x_true=[],re=[]):
         self.name    = pre +'CG'
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

# Conjugate gradient method preconditioned and deflated          
class DPCGS:
     def __init__(self,x,Its,tu,t,tp,rr,r_true,G,pre, dv,x_true=[],re=[]):
         self.name    = 'D'+pre+'CG'#, Z = ' + str(dv) + ' '
         self.marker  = '<'
         self.x       = x
         self.t_unit  = tu
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

# Conjugate gradient method deflated
class DCGS:
     def __init__(self,x,Its,t,tp,rr,r_true,G, dv,x_true=[],re=[]):
         self.name    = 'DCG'#', Z = '+ str(dv)
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


#%% Método de gradiente conjugado 
def CG(a,b,x_0,x_true,G,MaxIter,tol):
    t_In=0
    t_It=0 
    t_unit_f=0
    t_In_p=0
    t_It_p=0
    # ------------------------ Inicialización--------------------------------
    # Cálculo del tiempo de inicialización del método, se inicializa el tiempo
#    t_In = time.process_time()
    t_In_I = time.time()  
    t_In_I_p = time.process_time()

    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    r = b - A.dot(x)
    p = r.copy()
    

    # Compute the relative residual and put it in a list
    rr_vec = [] 
    r_true = r
    rr = LA.norm(r_true)/LA.norm(b)
    rr_vec.append(rr) 
    
    e_vec = []  
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
            
    
    # Make a list with the iterations
    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    # Cálculo del tiempo inicial
    #tt_In = time.process_time() - t_In
    #-------------------------------------------------------------------------
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    # -------------------------- Solución------------------------------------
    # Cálculo del tiempo de solución del sistema lineal, se inicializa el tiempo

    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    # Método de solución 
    while rr > tol and Its < MaxIter:
        
        x_old = x.copy()
        r_old = r.copy()
        p_old = p.copy()
        if Its == 0:
            t_unit = time.time()
#        w = np.dot(A,p_old)
        w = A.dot(p_old)
        alpha = np.dot(r_old,r_old)/np.dot(w,p_old)
        x = x_old + alpha*p_old

        r = r_old - alpha*w   
        beta = np.dot(r,r)/np.dot(r_old,r_old)
        p = r + beta*p_old 
        if Its == 0:
            t_unit_f = time.time()-t_unit
            

        r = b - A.dot(x)
        rr = LA.norm(r)/LA.norm(b)
        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its)
        
        x_true = SLA.spsolve(A,b)
        re = LA.norm(x_true-x)/LA.norm(x_true)
        e_vec.append(re)

    # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p

    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)
    CG_r = CGS(x,Its_vec,t_vec,t_vec_p,rr_vec,r_true,G,x_true,e_vec)
    return CG_r
#%%
def PGC(a,b,x_0,x_true,G,MaxIter,pre,tol):
    t_In=0
    t_It=0 
    t_unit_f=0
    t_In_p=0
    t_It_p=0
    # ------------------------ Inicialización--------------------------------
    # Cálculo del tiempo de inicialización del método, se inicializa el tiempo
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    A = a.copy()
    b = b.copy()
    x = x_0.copy()

    d = 1/A.diagonal()
    M1 = spdiags(d,0,G.N,G.N)
    r = b - A.dot(x)

    y  = M1*r
    ry = np.dot(r,y)
    p = y
    Mb  = M1*b

    # Compute the relative residual and put it in a list
    rr_vec = [] 
    r_true = r
    rr = LA.norm(r)/LA.norm(b)
    rr_vec.append(rr) 
    
    e_vec = []  
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
            
   
    # Make a list with the iterations
    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    
    # Cálculo del tiempo inicial
    #tt_In = time.process_time() - t_In
    #-------------------------------------------------------------------------
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
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
        y_old = y.copy()
        ry_old = ry.copy()
        p_old = p.copy()
        if Its == 0:
            t_unit = time.time()
        w = A.dot(p_old)
        alpha = ry_old/np.dot(p_old,w)
        x = x_old + alpha*p_old
        r = r_old - alpha*w
        #y = SLA.spsolve(M,r)
        y  = M1*r
        ry = np.dot(r,y)
        #z = SLA.spsolve(M2,z)
        beta = ry/ry_old
        p = y + beta*p_old 
        if Its == 0:
            t_unit_f = time.time()-t_unit
            
        #r_true = b - A.dot(x)
        #rr = LA.norm(y)/LA.norm(Mb)
        r = b - A.dot(x)
        rr = LA.norm(r)/LA.norm(b)
        re = LA.norm(x_true-x)/LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its) 
        

    # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p

    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)
    #print('&& PCG & ', LA.norm(r_true)/LA.norm(b))
    PCG_r = PCGS(x,Its_vec,t_vec,t_vec_p,rr_vec,r_true,G,pre,x_true,e_vec)

    return PCG_r
#%%

def DPGC(a,b,x_0,x_true,G,MaxIter,pre,tol,dv):
    t_In=0
    t_It=0 
    t_unit_f=0
    t_In_p=0
    t_It_p=0
    # ------------------------ Inicialización--------------------------------
    # Cálculo del tiempo de inicialización del método, se inicializa el tiempo
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    A = a.copy()

    b = b.copy()
    x = x_0.copy()

    d = 1/A.diagonal()
    M1 = spdiags(d,0,G.N,G.N)


    Z = G.Z
    ZT = Z.transpose()
    # Deflation matrices
    V = A*Z;
    V  = csc_matrix(V)
    E  = Z.transpose()*V
    E  = csc_matrix(E)
    EI = SLA.inv(E)   
    B  = V*EI;
    B  = csc_matrix(B)
    del d,V,E
    # Initializate vars
    r = b - A.dot(x)
    rr = LA.norm(r)/LA.norm(b)
    r = Deflation_P(B,Z,r)

    y = M1*r
    p_n = y

    Pb = Deflation_P(B,Z,b)

        # Compute the relative residual and put it in a list
    rr_vec = [] 

    rr_vec.append(rr) 

    e_vec = []  
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
            
   
    # Make a list with the iterations
    Its_vec = []
    Its = 0
    x_it =0
    Its_vec.append(Its)   
    ry = np.dot(r,y)
    
    # Cálculo del tiempo inicial
    #tt_In = time.process_time() - t_In
    #-------------------------------------------------------------------------
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
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
        y_old = y.copy()
        p_old = p_n.copy()
        ry_old = ry.copy()
        #w = Deflation_P(B,Z,A.dot(p_old))
        
        
        
        if Its == 0:
            t_unit = time.time()
        w  =  A*p_old# A.dot(p_old)
        w = w- B.dot(ZT.dot(w))
        alpha = ry_old/np.dot(p_old,w)
        x = x_old + alpha*p_old
        r = r_old - alpha*w
        #rr = LA.norm(r)/LA.norm(b)
        y = M1*r
        ry = np.dot(r,y)
        beta = ry/ry_old
        p_n = y + beta*p_old 
        if Its == 0:
            t_unit_f = time.time()-t_unit

        
        x_it  = Correction_Q(Z,EI,b) + Deflation_P(B,Z,x,T=True)
        r = b - A.dot(x_it)
        rr = LA.norm(r)/LA.norm(b)
        re = LA.norm(x_true-x_it)/LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        
        rr_vec.append(rr)
        Its_vec.append(Its) 
          
            # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p

    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    x_it  = Correction_Q(Z,EI,b) + Deflation_P(B,Z,x,T=True)
    r_true = b - A.dot(x_it)
    #print('&& DPCG &', LA.norm(r_true)/LA.norm(b))
   
    DPCG_r = DPCGS(x_it,Its_vec,t_unit_f,t_vec,t_vec_p,rr_vec,r_true,G,pre,dv,x_true,e_vec)

    return DPCG_r

#%%
def DGC(a,b,x_0,x_true,G,MaxIter,tol,dv):
    t_In=0
    t_It=0 
    t_unit_f=0
    t_In_p=0
    t_It_p=0
    # ------------------------ Inicialización--------------------------------
    # Cálculo del tiempo de inicialización del método, se inicializa el tiempo
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    x_it = x_0.copy
    # Deflation matrices

    Z = G.Z
    ZT = Z.T
    # Deflation matrices
    V = A*Z;
    V  = csc_matrix(V)
    E  = Z.transpose()*V
    E  = csc_matrix(E)
    EI = SLA.inv(E)
    B  = V*EI;
    B  = csc_matrix(B)
    # Initializate vars
    r = b - A.dot(x)
    rr = LA.norm(r)/LA.norm(b)
    r = Deflation_P(B,Z,r)
    Pb = Deflation_P(B,Z,b)
    p_n = r.copy()

    # Compute the relative residual and put it in a list
    rr_vec = [] 
    rr_vec.append(rr) 
    

    e_vec = []  
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
            
    # Make a list with the iterations
    Its_vec = []
    Its = 0
    x_it =0
    Its_vec.append(Its)   
    
    
    # Cálculo del tiempo inicial
    #tt_In = time.process_time() - t_In
    #-------------------------------------------------------------------------
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
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
        p_old = p_n.copy()
     
        if Its == 0:
            t_unit = time.time() 
        w  =  A*p_old# A.dot(p_old)
        w = w- B.dot(ZT.dot(w))
        alpha = np.dot(r_old,r_old)/np.dot(w,p_old)
        x = x_old + alpha*p_old
        r = r_old - alpha*w
        #rr = LA.norm(r)/LA.norm(b)
        beta = np.dot(r,r)/np.dot(r_old,r_old)
        p_n = r + beta*p_old 
        if Its == 0:
            t_unit_f = time.time()-t_unit
        #x_it  = Correction_Q(Z,EI,b) + Deflation_P(B,Z,x,T=True)
        #r_true = b - A.dot(x_it)
        #rr = LA.norm(r_true)/LA.norm(b)        

        
        x_it  = Correction_Q(Z,EI,b) + Deflation_P(B,Z,x,T=True)
        r =  b - A.dot(x_it)
        rr = LA.norm(r)/LA.norm(b)
        re = LA.norm(x_true-x_it)/LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its) 
        
    # Cálculo del tiempo total
    t_It_F = time.time()
    t_It   = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p   = t_It_I_p - t_It_F_p

    # Las variables de salida son la solución x, el número de iteraciones,
    # el residuo en cada iteración, el tiempo de inicialización, 
    # el tiempo que tarda el método en obtener la solución y el número de 
    # operaciones
    t_vec   = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    x_it  = Correction_Q(Z,EI,b) + Deflation_P(B,Z,x,T=True)
    r_true = b - A.dot(x_it)
    #print('&& DCG &', LA.norm(r_true)/LA.norm(b))

    DCG_r = DCGS(x_it,Its_vec,t_vec,t_vec_p,rr_vec,r_true,G,dv,x_true,e_vec)

    return DCG_r
