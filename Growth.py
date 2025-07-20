# -*- coding: utf-8 -*-
"""
Impactor_Design.py

Script for design of an aerosol impactor. Following "Design of Round-Nozzle Inertial Impactors Review with Updated
Design Parameters, F. J. Romay, E. García-Ruiz, Aerosol and Air Quality Research 2023"

Created 2025-02-04
@written by Kevin Maier (kevin.r.maier@tum.de)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gamma = 0.0727 # kg/s^2
M_w = 18.02/1000 # kg/mol
M_s = 58.44/1000 # kg/mol
ions = 2
m_s = 4*10**-20 # kg
roh_w = 1000 # kg/m^3
R = 8.31 # (kg*m^2)/(K*mol*s^2)
k_v = 0.59801 # W/m*K
H = 2453.5*M_w*1000 # J/mol at 20°C
D_v = 2.4*10**-5 # m^2/s
roh_p = 1000 # kg/m^3
ld = 66*10**(-9)
# 0.75 kg/m^3 water vapor density
g = 9.81 # m/s^2

def kelvin_ratio(d_p, T, ions):
    """"""
    K_r = ((1+((6*ions*m_s*M_w)/(M_s*roh_w*np.pi*d_p**3)))**-1)*math.exp((4*gamma*M_w)/(roh_w*R*T*d_p))
    return K_r

def equilibrium_droplet_T(T_inf, S_r):
    """T in °C"""
    equ_d_T = (((6.65+0.345*T_inf+0.0031*T_inf**2)*(S_r-1))/(1+(0.082+0.00782*T_inf)*S_r))
    return equ_d_T

def dV_inv(T_d, T_inf):
    dV_inv = ((T_d-T_inf)*R*k_v)/(D_v*M_w*H)
    return dV_inv

def droplet_lifetime(d_p):
    """for dp > 1 mum"""
    t = (R*roh_p*d_p**2)/8*D_v*M_w*(dV_inv)
    return t

def fuchs_factor(d_p):
    phi = (2*ld+d_p)/(d_p+5.33*(ld**2/d_p)+3.42*ld)
    return phi

def froude(u, L):
    """"""
    Fr = u/(g*L)**(1/2)
    return Fr

if __name__ == "__main__":
    # for k in np.arange(273.15, 323.15, 1):
    #     K_r = kelvin_ratio(0.02*10**-6, k, ions)
    #     print(f"T = {k}; K_r = {K_r}")

    # for k in np.linspace(0, 10, 1000):
    #      K_r = kelvin_ratio(k*10**-6, 293.15, 0)
    #      print(f"size / mum = {k}; K_r = {K_r}")

    print("hi")
