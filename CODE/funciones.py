# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:14:02 2019

@author: 


"""

import numpy as np
import math
#from plots import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as LA
import time
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt

from Clases import *
#%%    For loops
def vec_to_mat(p,Nx,Ny):
    ap=np.zeros((Nx,Ny))
    for j in range(0,Ny):
        for i in range(0,Nx):
            h=i+j*Nx
            ap[i,j]=p[h]
    return ap



def mat_to_vect(A,Nx,Ny):
    vA=np.zeros(Nx*Ny)
    for j in range(0,Ny):
        for i in range(0,Nx):
            h=i+j*Nx
            vA[h]=A[i,j]
    return vA  



def lamda_av3D(y,G):
    yx_av      = np.zeros(G.N)
    yy_av      = np.zeros(G.N)
    yz_av      = np.zeros(G.N)
    #y_av[0]   = y[0]
    #y_av[N]   = y[N-1]
    yx_av[0:G.N-1] = 2/((1/y[0:G.N-1]) + (1/y[1:G.N])) 
    yy_av[0:G.N-G.Nx] = 2/((1/y[0:G.N-G.Nx])+(1/y[G.Nx:G.N]))
    yz_av[0:G.N-G.Nl] = 2/((1/y[0:G.N-G.Nl])+(1/y[G.Nl:G.N])) 
    return(yx_av, yy_av, yz_av)

#%%
def Trans_FV3D (yx_av, yy_av,yz_av, G):    
   
    Tx = np.zeros(G.N) 
    Ty = np.zeros(G.N)
    Tz = np.zeros(G.N)

    
    Tx = G.Ax*yx_av/G.dxc
    Ty = G.Ay*yy_av/G.dyc
    Tz = G.Az*yz_av/G.dzc
    return (Tx, Ty, Tz)


#%%
def Hom(k_0,G):
    k = k_0*np.ones(G.N)
    Nlayers = 4
    slice_lay = int(G.Ny/Nlayers)
    Z = lil_matrix((G.N,4))
    Z[0*G.Nx *slice_lay : G.Nx*slice_lay,0] = 1
    Z[G.Nx *slice_lay : G.Nx*slice_lay*(2),1] = 1
    Z[2*G.Nx *slice_lay : G.Nx*slice_lay*(3),2] = 1       
    Z[3*G.Nx *slice_lay:,3] = 1
    return k,Z


def L(k_0,k_1,lay,G):
    k = k_0*np.ones(G.N)
    
    Nlayers = lay
    Z = lil_matrix((G.N,Nlayers)) 
    slice_lay = int(G.Ny/Nlayers)

    if np.mod(Nlayers,2)==0:
        for j in range(0, Nlayers):
            if np.mod(j,2)==0:
               k[j*G.Nx *slice_lay : G.Nx*slice_lay*(j+1)] = k_1
    for i in range(0, Nlayers):          
	        Z[i*G.Nx *slice_lay :(i+1)*G.Nx*slice_lay,i] = 1

               
    if np.mod(Nlayers,2)==1:
        for j in range(0, Nlayers):
            if np.mod(j,2)==0:
                k[j*G.Nx *slice_lay : G.Nx*slice_lay*(j+1)] = k_1
        Z[0*G.Nx *slice_lay : G.Nx*slice_lay,0] = 1
        Z[2*G.Nx *slice_lay : G.Nx*slice_lay*(3),1] = 1
        Z[G.Nx *slice_lay : G.Nx*slice_lay*(2),2] = 1
        Z[3*G.Nx *slice_lay:,3] = 1
    return k,Z

def FCP(k_0,k_1,G):
    k = k_0*np.ones(G.N)
    Z = lil_matrix((G.N,4)) 
    cx=int((G.Nx)/2)
    cy=int((G.Ny)/2)
    k1=1
    k2=250
    k3=500
    k4=1000
    for j in range (0, cy):
        k[j*(G.Nx): cx+(j*(G.Nx))] = k_1
        k[cx+j*(G.Nx): (j+1)*(G.Nx)] = k_0
        Z[j*(G.Nx): cx+(j*(G.Nx)),0] = 1
        Z[cx+j*(G.Nx): (j+1)*(G.Nx),1] = 1
        
    for j in range (cy, G.Ny):
        k[j*(G.Nx): cx+(j*(G.Nx))] = k_1
        k[cx+j*(G.Nx): (j+1)*(G.Nx)] = k_0
        Z[j*(G.Nx): cx+(j*(G.Nx)),2] = 1
        Z[cx+j*(G.Nx): (j+1)*(G.Nx),3] = 1
    return k,Z



def Lfrac_new (kh, kl, G, Nfrac):
    #N = Nx*Ny
    kperm = kl*np.ones(G.N)
    colz= 2*Nfrac
    Z = lil_matrix((G.N,colz)) 
    
    for i in range (0,G.N):
        if i% (G.Nx*(G.Ny/Nfrac))==0:
            kperm[i:i+G.Nx]=kh
            #print(i,i)
            
    
    hf=np.zeros(Nfrac)
    hc=0
    for i in range (0,G.N):   
        if i% (G.Nx*(G.Ny/Nfrac))==0 :
            hf[hc] =i
            hc = hc+1
            
    
    jpar =np.arange(0,Nfrac*2, 2)
    
    for j in range(0,Nfrac):
        Z[int(hf[j]):int(hf[j])+G.Nx, jpar[j]]=1
                         
         
    
    hl = np.zeros(Nfrac)
    hl = hf+G.Nx
    
    jimpar =np.arange(1,Nfrac*2, 2)
    
    for j in range(0,Nfrac):
        Z[int(hl[j]):int(hl[j])+int((G.Ny/Nfrac-1)*G.Nx), jimpar[j]]=1
    
    return kperm, Z

def Lfrac_new3D (kh, kl, G, Nfrac):
    #N = Nx*Ny
    kperm = kl*np.ones(G.N)
    colz= 2*Nfrac
    Z = lil_matrix((G.N,colz)) 
    
    for i in range (0,G.N):
        if i% (G.Nl*(G.Nz/Nfrac))==0:
            kperm[i:i+G.Nl]=kh
            #print(i,i)
            
    
    hf=np.zeros(Nfrac)
    hc=0
    for i in range (0,G.N):   
        if i% (G.Nl*(G.Nz/Nfrac))==0 :
            hf[hc] =i
            hc = hc+1
            
    
    jpar =np.arange(0,Nfrac*2, 2)
    
    for j in range(0,Nfrac):
        Z[int(hf[j]):int(hf[j])+G.Nl, jpar[j]]=1
                         
         
    
    hl = np.zeros(Nfrac)
    hl = hf+G.Nl
    
    jimpar =np.arange(1,Nfrac*2, 2)
    
    for j in range(0,Nfrac):
        Z[int(hl[j]):int(hl[j])+int((G.Nz/Nfrac-1)*G.Nl), jimpar[j]]=1
    #print(Z.todense())
    #print(Nfrac)
    return kperm, Z
#%%
def Perm(G, caso, perm_coef, Nfrac, lay=[], v=[],Clay=[]):
    k_0 = perm_coef[0]
    k_1 = perm_coef[1]
    
    # Homogeneous case
    if caso == 'Hom':
        k,Z = Hom(k_0,G)
          
    elif caso == 'L_frac':
        if G.Nz == 1:
            k, Z = Lfrac_new (k_0, k_1, G, Nfrac)
        else:
            k, Z = Lfrac_new3D (k_0, k_1, G, Nfrac)

    
    return k, Z



#%%
def A_mat_FV_no_for_allB3D(Tx,Ty,Tz,G,y,bc):
    

    # Initializate timing
    t_0 = time.time()
    A = lil_matrix((G.N,G.N))    # Prealocate A 
    
    idx = np.arange(0,G.N)
    idxLN = np.array(np.where((idx % G.Nx) !=0))
    idxRN = np.array(np.where((idx % G.Nx) !=G.Nx-1))
    idxNN = np.array(np.where(idx % G.Nl<G.Nl-G.Nx))
    idxSN = np.array(np.where(idx % G.Nl>G.Nx-1))
    idxUN = np.array(np.where(idx < G.N-G.Nl))
    idxDN = np.array(np.where(idx > G.Nl-1))
    
    idxLB = np.array(np.where((idx % G.Nx) ==0))
    idxRB = np.array(np.where((idx % G.Nx) ==G.Nx-1))
    idxNB = np.array(np.where(idx % G.Nl>=G.Nl-G.Nx))
    idxSB = np.array(np.where(idx % G.Nl<=G.Nx-1))
    idxUB = np.array(np.where(idx >= G.N-G.Nl))
    idxDB = np.array(np.where(idx <= G.Nl-1))
    
    #Left Neighbour
    A[idxLN,idxLN]   =  Tx[idxLN-1]
    A[idxLN,idxLN-1] = -Tx[idxLN-1]    
    
    #Right Neighbour  
    A[idxRN,idxRN]   += Tx[idxRN]
    A[idxRN,idxRN+1] -= Tx[idxRN]  
    
    #North Neighbour
    A[idxNN,idxNN]      += Ty[idxNN]
    A[idxNN,idxNN+G.Nx] -= Ty[idxNN]  
    
    #South Neighbour
    A[idxSN,idxSN]      += Ty[idxSN-G.Nx]
    A[idxSN,idxSN-G.Nx] -= Ty[idxSN-G.Nx]   
    
    ##Uper Neighbour
    A[idxUN, idxUN] += Tz[idxUN]
    A[idxUN, idxUN+G.Nl] -= Tz[idxUN]
    
    ##Down Neighbour
    A[idxDN, idxDN] += Tz[idxDN-G.Nl]
    A[idxDN, idxDN-G.Nl] -= Tz[idxDN-G.Nl]
    

    # Left boundary
    if bc.BL_type == 'D':
        A[idxLB,idxLB]  += G.Ax[idxLB]*y[idxLB]/(G.xcmesh[0]-G.xfmesh[0])
        
    # Right boundary  
    if bc.BR_type == 'D':
        A[idxRB,idxRB]  += G.Ax[idxRB]*y[idxRB]/(G.xfmesh[G.Nx]-G.xcmesh[G.Nx-1])
        
    # North boundary    
    if bc.BN_type == 'D':
        A[idxNB,idxNB]  += G.Ay[idxNB]*y[idxNB]/(G.yfmesh[G.Ny]-G.xcmesh[G.Ny-1])
        
    # South boundary    
    if bc.BS_type == 'D':
        A[idxSB,idxSB]  += G.Ay[idxSB]*y[idxSB]/(G.ycmesh[0]-G.yfmesh[0])
        
    # Upper_boundary
    if bc.BU_type == 'D':
        A[idxUB,idxUB]  += G.Az[idxUB]*y[idxUB]/(G.zfmesh[G.Nz]-G.zcmesh[G.Nz-1])
    #Down boundary
    if bc.BD_type == 'D':
        A[idxDB,idxDB]  += G.Az[idxDB]*y[idxDB]/(G.zcmesh[0]-G.zfmesh[0])
        
    # Get the total time
    t_nf =  time.time() - t_0      
    
    return A



#%%
def bc_array_FV_no_for_allB3D(G,y, bc):  
    # Initializate rhs vector
    b   =  np.zeros(G.N)
    
    # Initializate timing
    t_0 = time.time()
    idx = np.arange(0,G.N)
    idxLB = np.array(np.where((idx % G.Nx) ==0))
    idxRB = np.array(np.where((idx % G.Nx) ==G.Nx-1))
    idxNB = np.array(np.where(idx % G.Nl>=G.Nl-G.Nx))
    idxSB = np.array(np.where(idx % G.Nl<=G.Nx-1))
    idxUB = np.array(np.where(idx >= G.N-G.Nl))
    idxDB = np.array(np.where(idx <= G.Nl-1))
    
    # Dirichlet boundary conditions 
    # Left boundary
    Num_Left_bc    = np.arange(np.shape(idxLB)[1])
    if bc.BL_type == 'D':
        b[idxLB] += bc.BL[Num_Left_bc]*G.Ax[idxLB]*y[idxLB]/(G.xcmesh[0]-G.xfmesh[0])
    else:
        b[idxLB] += bc.BL[Num_Left_bc]
        
    # Right boundary
    Num_Right_bc    = np.arange(np.shape(idxRB)[1])
    if bc.BR_type  == 'D':
        b[idxRB] += bc.BR[Num_Right_bc]*G.Ax[idxRB]*y[idxRB]/(G.xfmesh[G.Nx]-G.xcmesh[G.Nx-1])
    else:
        b[idxRB] += bc.BR[Num_Right_bc]   
        
    # North boundary
    Num_North_bc   = np.arange(np.shape(idxNB)[1])
    if bc.BN_type  == 'D':
        b[idxNB] += bc.BN[Num_North_bc]*G.Ay[idxNB]*y[idxNB]/(G.yfmesh[G.Ny]-G.xcmesh[G.Ny-1])
    else:
        b[idxNB] += bc.BN[Num_North_bc] 
        
    # South boundary
    Num_South_bc    = np.arange(np.shape(idxSB)[1])
    if bc.BS_type  == 'D':
        b[idxSB] += bc.BS[Num_South_bc]*G.Ay[idxSB]*y[idxSB]/(G.ycmesh[0]-G.yfmesh[0])
    else:
        b[idxSB] += bc.BS[Num_South_bc]  
        
    # Upper boundary
    Num_Up_bc   = np.arange(np.shape(idxUB)[1])
    #print(Num_Up_bc, 'Num_Up_bc')
    if bc.BU_type  == 'D':
        b[idxUB] += bc.BU[Num_Up_bc]*G.Az[idxUB]*y[idxUB]/(G.zfmesh[G.Nz]-G.zcmesh[G.Nz-1])
    else:
        b[idxUB] += bc.BU[Num_Up_bc] 
        
        
    # Down boundary
    Num_Down_bc   = np.arange(np.shape(idxDB)[1])
    #print(Num_Down_bc, 'Num_Down_bc')
    if bc.BD_type  == 'D':
        b[idxDB] += bc.BD[Num_Down_bc]*G.Az[idxDB]*y[idxDB]/(G.zcmesh[0]-G.zfmesh[0])
    else:
        b[idxDB] += bc.BD[Num_Down_bc]
        
    # Get the total time
    t_nf =  time.time() - t_0                        
    return b



def qwell(G,well,Lambda,CW,pw):
    # Initialization
    kw = []
    qw = np.zeros(G.N)
    w  = 0
    
    for i in range (0, len(well)):
        kw = kw + [well[i][0] + well[i][1]*G.Nx]
    kw = np.array(kw)

    for i in kw:
        qw[i] = CW[w]*pw[w]*Lambda[i]
        w     = w + 1
        
    return(qw)

def AI_mat_full_FV_3D(G,rock,fluid,bc):
    Lambda_0 = rock.perm/fluid.mu  
    Lambda =  Lambda_0    
    [Lambdax_av, Lambday_av, Lambdaz_av] = lamda_av3D(Lambda,G)
    [Tx, Ty, Tz] = Trans_FV3D (Lambdax_av, Lambday_av,Lambdaz_av, G)
    A    = A_mat_FV_no_for_allB3D(Tx,Ty,Tz,G,Lambda,bc)
    q_bc = bc_array_FV_no_for_allB3D(G,Lambda, bc) 
    return  A, q_bc, Tx, Ty, Tz


def mat_to_vect_new(A):
    col = A.shape[1]
    row = A.shape[0]
    vA  = np.zeros(col*row)
    for i in range(0,row):
        for j in range(0,col):       
            h     = col*(i) +j
            vA[h] = A[i,j]
    return vA  

def Tens_to_vect_new(A):
    layers = A.shape[0]
    col    = A.shape[2]
    row    = A.shape[1]
    Np     = col*row
    vA     = np.zeros(col*row*layers)
    for k in range (0,layers):
        for i in range(0,row):
            for j in range(0,col):       
                h     = col*(i) +j +Np*k
                vA[h] = A[k,i,j]
    return vA




def cell_f3D (xf, yf, zf, Nx, Ny, Nz):
    
    Gx_1 = np.arange(0,Nx)
    Gy_1 = np.arange(0,Ny)
    Gz_1 = np.arange(0,Nz)
    meshx = np.meshgrid(Gy_1, Gz_1, xf)
    meshy = np.meshgrid(yf, Gz_1, Gx_1)
    meshz = np.meshgrid(Gy_1, zf, Gx_1)
    xf_vec = Tens_to_vect_new(meshx[2])
    yf_vec = Tens_to_vect_new(meshy[0])
    zf_vec = Tens_to_vect_new(meshz[1])
    return xf_vec, yf_vec, zf_vec


        
def cell_c3D (xf, yf, zf, Nx, Ny, Nz):
    xc = np.zeros(Nx)
    yc = np.zeros(Ny)
    zc = np.zeros(Nz)
    Nl = Nx*Ny
    N  = Nl*Nz
    
    for i in range(0, Nx):
        xc[i] = xf[i]+(xf[i+1]-xf[i])/2 
    hy=0
    for i in range(0, Nl, Nx):
        yc[hy] = yf[i]+(yf[i+Nx]-yf[i])/2 
        hy=hy+1
    hz=0
    for i in range(0, N, Nl):
        zc[hz] = zf[i]+(zf[i+Nl]-zf[i])/2 
        hz=hz+1
        
    mesh = np.meshgrid(yc, zc, xc)
    xc_vec = Tens_to_vect_new(mesh[2])
    yc_vec = Tens_to_vect_new(mesh[0])
    zc_vec = Tens_to_vect_new(mesh[1])
    
    return xc_vec, yc_vec, zc_vec


def xyz_mesh (xc_v, yc_v, zc_v, Nx, Ny, Nz, N):
    xc_mesh = np.zeros(Nx)
    yc_mesh = np.zeros(Ny)
    zc_mesh = np.zeros(Nz)
    xc_mesh = xc_v[0:Nx]
    yc_mesh = yc_v[0:Nx*Ny:Nx]
    zc_mesh = zc_v[0:N:Nx*Ny]
    return xc_mesh, yc_mesh, zc_mesh

def dxcenter(xc, Nx,Ny, N):
    dx= np.ones(N)
    for i in range(0, N-1):
        dx[i]=xc[i+1]-xc[i]
    return dx
     
def dycenter(y, Nx, Ny, N):
    dy= np.ones(N)
    Nl = Nx*Ny
    for i in range(0, N-Nx):
        dy[i]=y[i+Nx]-y[i]
    return dy   
    
def dzcenter(z, Nx, Ny, N):
    dz= np.ones(N)
    Nl = Nx*Ny
    for i in range(0, N-Nl):
        dz[i]=z[i+Nl]-z[i]
    return dz

def dxface(xf, Nx,Ny,Nz, N):
    dxf= np.zeros(N)
    hx=0
    Nxf = (Nx+1)*Ny*Nz
    for i in range(0, Nxf-1):
        if i%(Nx+1)!=Nx:
            dxf[hx]=xf[i+1]-xf[i]
            hx=hx+1
    return dxf
     
def dyface(yf, Nx, Ny, Nz, N):
    dyf= np.zeros(N)
    Nl = Nx*Ny
    Nyf = Nx*(Ny+1)*Nz
    Ny2D = Nx*(Ny+1)
    hy=0
    for i in range(0, Nyf-Nx):
        if i % (Ny2D)< Nl:
            #print(i, 'i')
            dyf[hy]=yf[i+Nx]-yf[i]
            hy=hy+1
    return dyf   
    

def dzface(zf, Nx, Ny,Nz, N):
    dzf= np.zeros(N)
    Nl = Nx*Ny
    Nzf = Nx*Ny*(Nz+1)
    hf=0
    for i in range(0, Nzf-Nl):
        dzf[hf]=zf[i+Nl]-zf[i]
        hf=hf+1
    return dzf


def Area_x(dyf, dzf, N):
    Ax = np.zeros(N)
    for i in range(0,N):
        Ax[i] = dyf[i]*dzf[i]
    return Ax

def Area_y(dxf, dzf, N):
    Ay = np.zeros(N)
    for i in range(0,N):
        Ay[i] = dxf[i]*dzf[i]
    return Ay

def Area_z(dxf, dyf, N):
    Az = np.zeros(N)
    for i in range(0,N):
        Az[i] = dxf[i]*dyf[i]
    return Az


def yfracture (Ny, Nfrac, Ly): 
    
    yfrac_init = Ly/Nfrac
    yfvec  = np.zeros(Ny+1)
    dy_frac = 0.01
    
    dy_lay = (Ly/Nfrac-dy_frac)/(Ny/Nfrac -1) 
    h=0
    for i in range(0, Ny+1):
        
        if i % int(Ny/Nfrac) == 0:
            #print(i,'i')
            #print(h, 'h')
            yfvec[i] = h*(Ly/Nfrac)
            h=h+1
        if i % int(Ny/Nfrac) == 1:
            yfvec[i]= yfvec[i-1]+dy_frac
        
    if Ny/Nfrac > 2:
    
        for i in range(0, Ny+1):
            if i % int(Ny/Nfrac) != 0 and i % int(Ny/Nfrac) != 1:
                yfvec[i] = yfvec[i-1]+dy_lay
    
    return yfvec

