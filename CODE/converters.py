#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:26:25 2020

@author: luisantonio
"""
import warnings

##This module converts the units of the input variables to the units used in the simulator
#%% Length
def feet2meter(ft):
    mts=0.3048*ft
    return(mts)
def cm2meter(cm):
    mts=0.01*cm
    return(mts)
    
def meter2feet(mts):
    ft=3.28084*mts
    return(ft)
    
def meter2cm(mts):
    cm=100*mts
    return(cm)
    
def Length_converter(L,units):
    if  units == 'ft':
        # Converts ft to meter
        L = feet2meter(L)
    elif units == 'cm':
        # Converts cm to meter
        L = cm2meter(L)
    elif units == 'm':
        # Does nothing
        print('')
    else:      
        warnings.warn('The length selected units are not allowed')
    return L


#%% Pressure
def psi2Pa(psi):
    Pa = 6894.76*psi
    return(Pa)

def Pa2psi(Pa):
    psi = 0.000145038*Pa
    return(psi)
    
def atm2Pa(atm):
    Pa = 101325*atm
    return(Pa)
    
def Pa2atm(Pa):
    atm = Pa*9.8692e-6
    return(atm)   
    
def Pressure_converter(P,units):
    if  units == 'psi':
        # Converts psi to Pa
        P = psi2Pa(P)
#        print('The Pressure units in Pa are: ', P)
    elif units == 'atm': 
        # Converts atm to Pa
        P = atm2Pa(P)
#        print('The Pressure units in Pa are: ', P)
    elif units == 'Pa': 
        print('')
        # Does nothing
#        print('The Pressure units in Pa are: ', P)    
    else:   
        warnings.warn('The pressure selected units are not allowed')
#        print('The Pressure selected units are not allowed')
    return P

#%% Density
def lbft2kgm(lbft3):
    kgm3 = (3.28084**3)/2.20462*lbft3 
    return(kgm3)

def kgm2lbft(kgm3):
    lbft3 = (2.20462)/(3.28084**3)*kgm3 
    return(lbft3) 
def Density_converter(rho,units):
    if  units == 'lbft3':
        # Converts lbft to kgm3
        rho = lbft2kgm(rho)
#        print('The Density units in kgm3 are: ', rho)
    elif units == 'kgm3': 
        # Does nothing
        print('')
#        print('The Density units in kgm3 are: ', rho)    
    else:   
        warnings.warn('The density selected units are not allowed')     
#        print('The Density selected units are not allowed')
    return rho
    
#%% Permeability   
def Da2m2(Da):
    m2= Da*9.869233*10**(-13)
    return(m2) 
def m22mDa(m2):
    Da= m2*1013249965828.145
    return(Da)
    
def Permeability_converter(K,units):
    if  units == 'Da':
        # Converts Da to m2
        K = Da2m2(K)
    elif units == 'm2': 
        # Does nothing
        print('')
    else:   
        warnings.warn('The density selected units are not allowed')     
    return K
    
#%% Viscosity
def cp2Pas(cp):
    Pas = 0.001*cp
    return(Pas)
def Pas2cp(Pas):
    cp = 1000*Pas
    return(cp)
def Viscosity_converter(mu,units):
    if  units == 'cp':
        # Converts cp to Pas
        mu = cp2Pas(mu)
    elif units == 'Pas': 
        # Does nothing
        print('')
    else:   
        warnings.warn('The viscosity selected units are not allowed')
    return mu
    
#%% Flowrate

def stbday2ms(stbday):
    m3s= 6.28*86400*stbday
    return(m3s)

def ms2stbday(m3s):
    stbday = m3s/(6.28*86400)
    return(stbday)
    
def Flux_converter(q,units):
    if  units == 'stbday':
        # Converts cp to Pas
        q = stbday2ms(q)
    elif units == 'm3s': 
        # Does nothing
        print('')
    else:      
        warnings.warn('The flux selected units are not allowed')
    return q
   