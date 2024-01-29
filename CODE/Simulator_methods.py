# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:59:37 2023

@author: gbdiaz
"""


# ============================================================================
# =========================== Import libraries ================================ 
# ============================================================================
# Python functions/libraries
import os
import sys
import json
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import linalg as SLA

# Own libraries
from CGM import *
from MGM import *
from Clases import *
from funciones import *

# =============================================================================
# ------------------------------- Solvers -------------------------------------
# =============================================================================
# Preconditioner
M_1 ='J'  # Jacobi

# MG smoother iterarions
S1_it = 10
S2_it = 10

# Deflation vectors (subdomain or eigenvectors)
def_vec = ['SD','Eigs']
dv = def_vec[0]

# Initialization, options: "random","DCG","DPCG"
x_init = "random"

# Stopping criteria
tol       = 5e-8
MaxIter   = 10000

# =============================================================================
# ------------------------------- Reservoir -----------------------------------
# =============================================================================
# -----------------------------------------------------------------------------
# -------------------------------   Rock   ------------------------------------
# -----------------------------------------------------------------------------
# -------------------------  Permeability field -------------------------------
# -----------------------------------------------------------------------------
# Case can be Hom or L_frac (fractures)
c_case = "Hom"
# If a fractured case is selected, the first permeability is for the reservoir, 
# and the second for the fractyres or clays
perm = [0.001,0.01] 
K_units = 'Da'
Nfrac = 2
# -----------------------------------------------------------------------------
#--------------------------  Porosity field -----------------------------------
# -----------------------------------------------------------------------------
poro = 0.25 
# -----------------------------------------------------------------------------
# --------------------------------- Fluid -------------------------------------
# -----------------------------------------------------------------------------
# -------------------------------- Density ------------------------------------
# -----------------------------------------------------------------------------
rho = 1
rho_units = 'lbft3'
# -----------------------------------------------------------------------------
# -------------------------------- Viscosity ----------------------------------
# -----------------------------------------------------------------------------
mu = 0.51 
mu_units = 'cp'
# -----------------------------------------------------------------------------
# --------------------------------   Dimensions  ------------------------------
# -----------------------------------------------------------------------------
Nx = 32
Ny = Nx
Nz = 1
N=Nx*Ny*Nz

Lx = 762
Ly = 762  
Lz = 762
L_units = 'm'

# -----------------------------------------------------------------------------
# --------------------------------  Boundary conditions   ---------------------
# -----------------------------------------------------------------------------

BL_type  = 'N'
BL_value = 0
BL_units = 'stbday'

BR_type  = 'N'
BR_value = 0
BR_units ='stbday' 

BN_type  = 'D'
BN_value = 8000 
BN_units = 'psi'

BS_type  = 'D'
BS_value = 0
BS_units = 'psi'

BU_type  = 'D'
BU_value = 8000
BU_units = 'psi'

BD_type  = 'D'
BD_value = 0
BD_units ='psi'
# =============================================================================
#--------------------------------- Simulation  -------------------------------
# =============================================================================

if c_case == "Hom":
    perm_coef = [perm[0],perm[0]]
    layers = 0
    Nfrac = 0
else:
    perm_coef = [perm[0],perm[1]]
    layers = int(Nfrac*2)

# ===========++++==============================================================
# -------------------------- Construct reservoir  -----------------------------
# ==============++++===========================================================

# Construction of the grid

Cx = np.linspace(0, Lx, Nx+1)
Cz = np.linspace(0,Lz,Nz+1)
if c_case == "Hom":
    Cy = np.linspace(0, Ly, Ny+1)
else:
    Cy = yfracture (Ny, Nfrac, Ly)
        

G = Grid_New3D(Lx, Ly, Lz, Nx, Ny, Nz, N, Cx, Cy, Cz, L_units)
 
# Boundary conditions initialization
bc  = BC_3D(BL_value*np.ones(G.N), BR_value*np.ones(G.N), 
         BN_value*np.ones(G.N), BS_value*np.ones(G.N),
         BU_value*np.ones(G.N), BD_value*np.ones(G.N),
         BL_units, BR_units, BN_units, BS_units, BU_units, BD_units, 
         BL_type, BR_type, BN_type, BS_type, BU_type, BD_type)   

Fluid_c = {'rho':[rho* np.ones(G.N) ,rho_units],'mu':[mu* np.ones(G.N) ,mu_units]}
fluid   = Class_Fluid(Fluid_c)

K,G.Z   = Perm(G, c_case, perm_coef, Nfrac, layers)
rock_c  = {'poro':[poro* np.ones(G.N)],'K':[K,K_units]}
rock    = Class_Rock(rock_c)

# Matrix construction
[A, b, Tx, Ty, Tz] = AI_mat_full_FV_3D(G,rock,fluid,bc)
A_S    = csc_matrix(A)  
x_true = SLA.spsolve(A_S, b)

# Initialization
if x_init == "random":
    x_0    = np.random.rand(np.size(b))   
elif x_init == "DCG":
    x_0    = np.random.rand(np.size(b)) 
    DCG_GD   =  DGC(A_S,b,x_0,x_true,G,1,tol,dv)
    x_0 = DCG_GD.x
elif x_init == "DPCG":
    x_0    = np.random.rand(np.size(b)) 
    DPCG_GD   =  DPGC(A_S,b,x_0,x_true,G,1,pre,tol,dv)
    x_0 = DPCG_GD.x
    
# Solution of the matrix with the different methods
PCG_sol   =  PGC(A_S,b,x_0,x_true,G,MaxIter,M_1,tol)
CG_sol    =  CG(A_S,b,x_0,x_true,G,MaxIter,tol)
DCG_sol   =  DGC(A_S,b,x_0,x_true,G,MaxIter,tol,dv)
DPCG_sol  =  DPGC(A_S,b,x_0,x_true,G,MaxIter,M_1,tol,dv)
MG_sol    =  MG(A_S,b,x_0,x_true,G,S1_it,S2_it, MaxIter,tol)
MGCG_sol  =  MGCG(A_S,b,x_0,x_true,G,MaxIter,tol,S1_it,S2_it)

#%%
# =============================================================================
# ----------------------------------- Plots  --------------------------------
# =============================================================================
# To select the method to plot just change the method name (met_name)
met_name = MG_sol

p = met_name.x

# Plot residuals
plt.plot(met_name.rres)
plt.xlabel('Iteration', fontsize=12) 
plt.ylabel('Relative residual', fontsize=12)
plt.yscale('log')
plt.title("Relative residual of "+met_name.name)
plt.show()

# Plot field
if Nz ==1:
    x = G.xcmesh  
    y = G.ycmesh 
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    Z = p.reshape((Nx,Ny)) # Convert array to matrix
    plt.pcolormesh(X, Y, Z) 
    plt.title("Solution of "+met_name.name)
    plt.show()



