#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# ============================================================================
# -----------------------  Librerias requeridas  -----------------------------
# ============================================================================
from converters import *
from funciones import *
import numpy as np


# ============================================================================
# ---------------------------- Dimensiones -----------------------------------
# ============================================================================

# Esta clase guarda todas las variables que estan relacionadas con las 
# dimensiones del yacimiento. Adicionalmente, transforma las dimensiones al SI
# =============================================================================

class Grid_New3D:
     def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, N, Cx, Cy, Cz, L_units):
         
         

         
         [xf_vec, yf_vec, zf_vec] = cell_f3D(Cx, Cy, Cz, Nx, Ny, Nz)
         [xc_vec, yc_vec, zc_vec] = cell_c3D (xf_vec, yf_vec, zf_vec, Nx, Ny, Nz)
         [xc_m, yc_m, zc_m] = xyz_mesh (xc_vec, yc_vec, zc_vec, Nx, Ny, Nz, N)
         dxc = dxcenter(xc_vec, Nx,Ny, N)
         dyc = dycenter(yc_vec, Nx,Ny, N)
         dzc = dzcenter(zc_vec, Nx,Ny, N)        
         dxf = dxface(xf_vec, Nx,Ny,Nz, N)
         dyf = dyface(yf_vec, Nx,Ny,Nz, N)
         dzf = dzface(zf_vec, Nx,Ny,Nz, N)
         Ax = Area_x(dyf, dzf, N)
         Ay = Area_y(dxf, dzf, N)
         Az = Area_z(dxf, dyf, N)
         
         self.Nx = Nx
         self.Ny = Ny
         self.Nz = Nz
         self.N  = Nx*Ny*Nz
         self.Nl = self.Nx*self.Ny 
         self.Lx = Length_converter(Lx,L_units)   
         self.Ly = Length_converter(Ly,L_units)
         self.Lz = Length_converter(Lz,L_units)
         self.xfmesh = Cx
         self.yfmesh = Cy
         self.zfmesh = Cz
         self.xcmesh = xc_m
         self.ycmesh = yc_m
         self.zcmesh = zc_m
         self.xf = xf_vec
         self.yf = yf_vec
         self.zf = zf_vec
         self.xc = xc_vec
         self.yc = yc_vec
         self.zc = zc_vec
         self.dxc = dxc
         self.dyc = dyc
         self.dzc = dzc
         self.dxf = dxf
         self.dyf = dyf
         self.dzf = dzf
         self.Ax = Ax
         self.Ay = Ay
         self.Az = Az
         self.L_units = 'm'
         
         

         
class Grid_NewMG:
     def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, Cx, Cy, L_units):
         
         
         [xf_vec, yf_vec, yf1_vec] = cell_f(Cx, Cy,Nx, Ny)
         [xc_vec, yc_vec] = cell_c(Cx, Cy,Nx, Ny)
         [dxf, dxc] = Gdx (xf_vec, xc_vec, Nx, Ny)
         [dyf, dyc] = Gdy (yf1_vec, yc_vec, Nx, Ny)
         
         self.Lx = Length_converter(Lx,L_units)   
         self.Ly = Length_converter(Ly,L_units)
         self.Lz = Length_converter(Lz,L_units)
         self.xfmesh = Cx
         self.yfmesh = Cy
         self.xf = xf_vec
         self.yf = yf_vec
         self.xc = xc_vec
         self.yc = yc_vec
         self.dxf = dxf
         self.dyf = dyf
         self.dxc = dxc
         self.dyc = dyc
         self.Nx = Nx+1
         self.Ny = Ny+1
         self.Nz = Nz
         self.Dx = Lx/Nx
         self.Dy = Ly/Ny
         self.Dz = Lz/Nz
         self.N  = Nx*Ny*Nz
         self.Vol = self.Dx*self.Dy*self.Dz 
         self.L_units = 'm'
              

# =============================================================================
# ============================================================================
# ----------------------------- Roca -----------------------------------------
# ============================================================================

# Esta clase guarda todas las variables que estan relacionadas con la roca
# del yacimiento. Adicionalmente, transforma las dimensiones al SI
class Rock:
     def __init__(self, K, K_units, poro):
         self.perm = Permeability_converter(K,K_units)
         self.perm_units = 'm2'
         self.poro  = poro

# Esta función crea la calse rock a partir de los datos de la lista de entrada 
# rock_c, que contiene todas las variables dadas por el usuario        
def Class_Rock(rock_c):
    poro     = rock_c['poro'][0]   # Porosidad    
    K        = rock_c['K'][0]        # Permeabilidad
    K_units  = rock_c['K'][1]        # Unidades de permeabilidad 

    # Creación de la clase con los parámetros dados
    rock = Rock(K, K_units, poro)
    return rock

# ============================================================================
# ------------------------------- Fluido -------------------------------------  
# ============================================================================

# Esta clase guarda todas las variables que estan relacionadas con el fluido
# del yacimiento. Adicionalmente, transforma las dimensiones al SI
class Fluid:
     def __init__(self,rho, rho_units, mu,mu_units) :
         self.rho = Density_converter(rho,rho_units)
         self.rho_units = 'kg/m3'
         self.mu  = Viscosity_converter(mu,mu_units)
         self.mu_units = 'Pa'


# Esta función crea la calse fluid a partir de los datos de la lista de entrada 
# Fluid_c, que contiene todas las variables dadas por el usuario             
def Class_Fluid(Fluid_c):
  
    rho       = Fluid_c['rho'][0]    # Densidad
    rho_units = Fluid_c['rho'][1]    # Unidades de densidad    
    mu       = Fluid_c['mu'][0]    # Viscosidad
    mu_units = Fluid_c['mu'][1]    # Unidades de viscosidad 

    # Creación de la clase con los parámetros dados
    fluid = Fluid(rho, rho_units, mu,mu_units)  
    return fluid 
# ============================================================================
#------------------------ Condiciones de frontera ----------------------------
# ============================================================================

# Esta clase guarda todas las variables que estan relacionadas con las 
# condiciones de frontera del yacimiento. Adicionalmente, transforma las 
# dimensiones al SI
class BC:
     def __init__(self, BL, BR, BN, BS, BL_units, BR_units, BN_units, 
                  BS_units, BL_type, BR_type, BN_type, BS_type):              
         if BL_type == 'D':
            self.BL = Pressure_converter(BL,BL_units)
            self.BL_units = 'Pa'            
         else: 
            self.BL = Flux_converter(BL,BL_units) 
            self.BL_units = 'm3s'            
         self.BL_type  = BL_type
         
         if BR_type == 'D':
            self.BR = Pressure_converter(BR,BR_units)
            self.BR_units = 'Pa'  
         else: 
            self.BR = Flux_converter(BR,BR_units) 
            self.BR_units = 'm3s' 
         self.BR_type  = BR_type
         
         if BN_type == 'D':
            self.BN = Pressure_converter(BN,BN_units)
            self.BN_units = 'Pa'
         else: 
            self.BN = Flux_converter(BN,BN_units) 
            self.BN_units = 'm3s'
         self.BN_type  = BN_type 
         
         if BS_type == 'D':
            self.BS = Pressure_converter(BS,BS_units)
            self.BS_units = 'Pa'
         else: 
            self.BS = Flux_converter(BS,BS_units)
            self.BS_units = 'm3s'
         self.BS_type  = BS_type
         
class BC_3D:
     def __init__(self, BL, BR, BN, BS, BU, BD, BL_units, BR_units, BN_units, 
                  BS_units, BU_units, BD_units, BL_type, BR_type, BN_type, BS_type, BU_type, BD_type):              
         if BL_type == 'D':
            self.BL = Pressure_converter(BL,BL_units)
            self.BL_units = 'Pa'            
         else: 
            self.BL = Flux_converter(BL,BL_units) 
            self.BL_units = 'm3s'            
         self.BL_type  = BL_type
         
         if BR_type == 'D':
            self.BR = Pressure_converter(BR,BR_units)
            self.BR_units = 'Pa'  
         else: 
            self.BR = Flux_converter(BR,BR_units) 
            self.BR_units = 'm3s' 
         self.BR_type  = BR_type
         
         if BN_type == 'D':
            self.BN = Pressure_converter(BN,BN_units)
            self.BN_units = 'Pa'
         else: 
            self.BN = Flux_converter(BN,BN_units) 
            self.BN_units = 'm3s'
         self.BN_type  = BN_type 
         
         if BS_type == 'D':
            self.BS = Pressure_converter(BS,BS_units)
            self.BS_units = 'Pa'
         else: 
            self.BS = Flux_converter(BS,BS_units)
            self.BS_units = 'm3s'
         self.BS_type  = BS_type
         
         if BU_type == 'D':
            self.BU = Pressure_converter(BU,BU_units)
            self.BU_units = 'Pa'
         else: 
            self.BU = Flux_converter(BU,BU_units)
            self.BU_units = 'm3s'
         self.BU_type  = BU_type
         
         if BD_type == 'D':
            self.BD = Pressure_converter(BD,BD_units)
            self.BD_units = 'Pa'
         else: 
            self.BD = Flux_converter(BD,BD_units)
            self.BD_units = 'm3s'
         self.BD_type  = BD_type
