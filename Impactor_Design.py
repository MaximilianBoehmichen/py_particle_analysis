# -*- coding: utf-8 -*-
"""
Impactor_Design.py

Script for design of an aerosol impactor. Following "Design of Round-Nozzle Inertial Impactors Review with Updated
Design Parameters, F. J. Romay, E. García-Ruiz, Aerosol and Air Quality Research 2023"

Created 2025-02-04
@written by Kevin Maier (kevin.r.maier@tum.de)
"""

import math
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
## geometry parameters
### nozzle diameter (mm)
W = 1
### jet-to-plate-distance (mm)
S = 1
### throat length
T = 1
### number of nozzler
n = 1
### cluster diameter (mm)
D_c = 20
### nozzle to nozzle distance (mm)
A = 10
### discharge parameter of nozzles - <1 for tapered between 0.7 and 0.9 for straight nozzles 0.6
C_d = 0.6

## particle parameters
### desired cut-off diameter (nm)
d_50 = 1000
### particle density (kg/m^3)
roh_p = 1000 # water = 1000

## gas parameters
### absolute gas viscosity (kg/(m*s) = Pa*s) ! f(T, p) ! -> build calculator
mu = 18.13*10**-6 # (at 20°C)
### flow rate (ccm/min)
Q = 300
### pressure (mbar)
p_g = 1013.25
### density (kg/m^3)
roh_g = 1.204

## natural constants
###gravitation (m/s^2)
g = 9.81


def convert_to_SI(W, S, T, D_c, A, d_50, Q, p_g):
    W_m = W * (10**-3)
    S_m = S * (10**-3)
    T_m = T * (10**-3)
    D_c_m = D_c * (10**-3)
    A_m = A * (10**-3)
    d_50_m = d_50 * (10**-9)
    Q_m3_s = Q * (10**-6) / 60  # (m^3/s)
    p_g_Pa = p_g * (10**2)
    return W_m, S_m, T_m, D_c_m, A_m, d_50_m, Q_m3_s, p_g_Pa


def mean_free_path(p_g_Pa):
    """calculates mean free path of air in m based on pressure from mfp at 20°C and 101 kPa as given in Aerosol
    Technology on p. 20"""
    mfp = (66*10**-9)*((101*10**3)/p_g_Pa)
    return mfp


def cunningham_slip(d_m, p_g_Pa):
    """cunningham slip from Aerosol Technology (2022) p. 42 with a=2.33, b=0.96, c=-0.50 pr p. 43 in pressure
    dependent form"""
    mfp = mean_free_path(p_g_Pa)
    # a = 2.33
    # b = 0.96
    # c = 0.50
    # C_c = 1 + (mfp/d_m) * (a + b*math.exp(-c*(d_m/mfp)))
    a = 15.60
    b = 7.00
    c = -0.059
    d_mu_m = d_m*10**6
    p_g_kPa = p_g_Pa*10**-3
    C_c = 1 + (1 / (d_mu_m*p_g_kPa)) * (a + b*math.exp( c*(d_mu_m*p_g_kPa))) # takes mu_m and kPa
    return C_c # wrong result atm


def nozzle_exit_velocity(Q_m3_s, n, W_m):
    """nozzle exit velocity in m/s"""
    V0 = (4*Q_m3_s) / (n*math.pi*W_m**2)
    return V0


def stokes(d_m, roh_p, V0, mu, W_m):
    """unit-less stokes number, describing ratio of characteristic time of particle to characteristic time of a flow"""
    C_c = cunningham_slip(d_m, p_g * (10**2))
    Stk = (C_c*d_m**2*roh_p*V0) / (9*mu*W_m)
    return Stk


def sqrt_stokes(Stk):
    """dimensionless particle diameter root of stokes"""
    Sqrt_Stk = math.sqrt(Stk)
    return Sqrt_Stk


def calc_d_50(C_c, roh_p, V0, mu, W_m, Stk):
    """restructured stokes"""
    d_50_m = math.sqrt((9*mu*W_m) / (C_c*roh_p*V0)) * math.sqrt(Stk)
    return d_50_m


def reynolds(roh_g, V0, W_m, mu):
    """unit-less reynolds number describing flow regime Re<2300 laminar, 2300<Re<2900 transition, 2900<Re turbulent"""
    Re = (roh_g * V0 * W_m) / mu
    return Re


def d_jet_plate(S_m,W_m):
    """jet to plate distance, typically 1-5"""
    S_W = S_m/W_m
    return S_W


def cross_flow_parameter(W_m, n, D_c_m):
    """cross-flow-parameter for multi-nozzle impactors, should be CFD < 1.2"""
    CFP = (W_m * n) / (2 * D_c_m)
    return CFP


def nozzle_spacing_parameter(A_m, W_m):
    """nozzle to nozzle distance ratio for multi-nozzle impactors, should be NSP > 4"""
    NSP = A_m / W_m
    return NSP


def froude(V0, g, W_m):
    """froude number descirbing ratio of flow inertia vs gravitational force for bigger particles"""
    Fr = V0**2 / (g*W_m)
    return Fr


def pressure_drop(roh_g, V0, C_d):
    """pressure drop across the impactor stage flow and density must be actual flow and density at pressure of stage"""
    dP = (roh_g*V0**2) / (2*C_d**2)
    return dP


def W_from_stokes(d_50_m, p_g_Pa, roh_p, roh_g, Re, Stk):
    """from Marple & Willeke 1974"""
    C_c = cunningham_slip(d_50_m, p_g_Pa)
    W_m = ((C_c**(1/2)) * d_50_m)*((roh_p * Re) / (9 * roh_g * Stk))**(1/2)
    return W_m


def Q_from_stokes(d_50_m, p_g_Pa, mu, n, roh_p, roh_g, Re, Stk):
    """from Marple & Willeke 1974"""
    C_c = cunningham_slip(d_50_m, p_g_Pa)
    Q_m3_s = (math.pi/12) * ((roh_p / Stk)**(1/2)) * ((Re / roh_g)**(3/2)) * n * mu * (C_c**(1/2)) * d_50_m
    Q_ccm = Q_m3_s * 10**6 * 60
    return Q_m3_s, Q_ccm


def calc_impactor(d_50, roh_p, Re, Stk, Q_ccm, n):  # For design, flow must be fixed Re must then be chosen
    W_m, S_m, T_m, D_c_m, A_m, d_50_m, Q_m3_s, p_g_Pa = convert_to_SI(1, S, T, D_c, A, d_50, Q_ccm, p_g)
    W = W_from_stokes(d_50_m, p_g_Pa, roh_p, roh_g, Re, Stk)
    # Q_m3_s, Q_ccm = Q_from_stokes(d_50_m, p_g_Pa, mu, n, roh_p, roh_g, Re, Stk)
    V0 = nozzle_exit_velocity(Q_m3_s, n, W)
    dp = pressure_drop(roh_g, V0, C_d)
    # Stk = stokes(d_50_m, roh_p, V0, mu, W)
    # Re = reynolds(roh_g, V0, W, mu)
    print(f"sqrt Stk = {math.sqrt(Stk)}, Re = {Re}, D_50 = {d_50}, roh = {roh_p}, n = {n}, "
          f"W / mm = {W*1000}, Q / L/min = {Q_ccm/1000}, V0 / m/s = {V0}, dp / mbar = {dp/100},")
    return W, Q_ccm, V0, dp


def d50_set_W_Q(W, D_c, A, Q, p_g, Stk):
    dummy_50 = 10000 # as cunningham is almost 1 for 10 micron particles
    d_50s, Ws, Qs, Vs, Res, dps = [],[],[],[],[],[]

    for k in Q:
        for i in W:
            W_m, S_m, T_m, D_c_m, A_m, d_50_m, Q_m3_s, p_g_Pa = convert_to_SI(i, i, i, D_c, A, dummy_50, k, p_g)

            V0 = nozzle_exit_velocity(Q_m3_s, n, W_m)
            Re = reynolds(roh_g, V0, W_m, mu)

            ct_list = []
            C_list = []
            tol_list= []
            d_50_list = []

            tol = 1
            ct = 0

            while abs(tol) > (10**-12):  # iterative calculation of dp and Cc of the
                C_c = cunningham_slip(d_50_m, p_g_Pa)
                d_50_m = calc_d_50(C_c, roh_p, V0, mu, W_m, Stk)
                tol = dummy_50 - d_50_m
                dummy_50 = d_50_m
                ct += 1
                # print(ct)
                ct_list.append(ct)
                C_list.append(C_c)
                tol_list.append(tol)
                d_50_list.append(d_50_m)

            fig, axs = plt.subplots(3)
            axs[0].scatter(ct_list, [i * (10 ** 9) for i in d_50_list])
            axs[0].set_ylabel("Particle Diameter / nm")
            axs[1].scatter(ct_list, C_list)
            axs[1].set_ylabel("Cunningham Correction Factor")
            axs[2].scatter(ct_list, tol_list)
            axs[2].set_xlabel("Iterations")
            axs[2].set_ylabel("Tolerance")
            plt.show()

            dp_mbar = pressure_drop(roh_g, V0, C_d) / 100

            d_50s.append(d_50_m*10**9)
            Ws.append(i)
            Qs.append(k)
            Vs.append(V0)
            Res.append(Re)
            dps.append(dp_mbar)

    return d_50s, Ws, Qs, Vs, Res, dps


def save_to_excel(dataframetosave):
    df = pd.DataFrame(dataframetosave)
    fileextention = input("add file extension")
    path = "I:/Muenchen/05_Analytics_Bioaerosole/Aerosol/Printed-Impactor/" + fileextention + ".xlsx"
    # sheet_name = input("add sheet name") # does not work like that, file must be open in writer
    df.to_excel(path)
    # df.to_excel(path, sheet_name=sheet_name)
    return


if __name__ == "__main__":

    n = [1, 3, 5, 7, 9, 11]### number of nozzles

    ### Other defining parameters
    Re = [2200]  # [500, 1500, 2200, 3000]
    Stk = [0.507]  # [0.489**2, 0.498**2, 0.507**2, 0.518**2]

    ## particle parameters
    d_50 = [1000] # , 2500, 10000] ### desired cut-off diameter (nm)
    roh_p = 1000  ### particle density (kg/m^3)

    Q_in = [100000]

    sqrt_Stks, Res, D_50s, rohs, ns, Ws, Qs, V0s, dps = [],[],[],[],[],[],[],[], []

    for j in range(len(Re)):
        for k in d_50:
            for i in n:
                W, Q_ccm, V0, dp = calc_impactor(k, roh_p, Re[j], Stk[j], Q_in[j], i)
                sqrt_Stks.append(math.sqrt(Stk[j]))
                Res.append(Re[j])
                D_50s.append(k)
                rohs.append(roh_p)
                ns.append(i)
                Ws.append(W*1000)
                Qs.append(Q_ccm/1000)
                V0s.append(V0)
                dps.append(dp/100)

    save_to_excel({"sqrt Stk": sqrt_Stks, "Re": Res, "D50": D_50s, "Particle Density kg/m^3": rohs, "n Nozzles": ns,
                   "Diameter Nozzle / mm": Ws, "Flow Rate / L/min": Qs, "Nozzle exit Velocity / m/s": V0s,
                   "Pressure Drop / mbar": dps})


    # ## geometry parameters
    # # W = 1 ### nozzle diameter (mm)
    # S = W ### jet-to-plate-distance (mm)
    # T = W ### throat length
    # A = 5*W ### nozzle to nozzle distance (mm) A/W>4
    # D_c = W*n/4  ### cluster diameter (for multiple nozzles) (mm) W*n/4*D_c <1.2 -> maybe mae it bigger?
    # C_d = 0.6 ### discharge parameter of nozzles - <1 for tapered between 0.7 and 0.9 for straight nozzles 0.6

    """choose the desired calculation here if you want"""

    # d_50s, Ws, Qs, Vs, Res, dps = Q_dp_set_d50_W(W, D_c, A, d_50, p_g, Stk)  # W and d_50 as list

    # d_50s, Ws, Qs, Vs, Res, dps = d50_set_W_Q(W, D_c, A, Q, p_g, Stk) # W and Q as list

    # d_50s, Ws, Qs, Vs, Res, dps = W_Re_set_d50_Q(d_50, D_c, A, Q, p_g, n, Stk)

    # calc_Stk = []
    # for k in range(len(d_50s)):
    #     d_50_m = d_50s[k] * (10 ** -9)
    #     C_c = cunningham_slip(d_50_m, p_g*(10**2))
    #     W_m = Ws[k]* (10**-3)
    #     Q_m3_s = Qs[k]* (10**-6) / 60  # (m^3/s)
    #     V0 = nozzle_exit_velocity(Q_m3_s, 1, W_m)
    #     calc_Stk.append(stokes(C_c, d_50_m, roh_p, V0, mu, W_m))
    #
    # save_to_excel(d_50s, Ws, Qs, Vs, Res, dps, calc_Stk)

    """the two directions do not give the same result unfortunately"""

# Sqrt of Stokes 0.489 for Re = 500, 0.498 for Re = 1500, 0.507 for Re 2200
# Re 500 to 3000
# S/W 1 to 5
# CFP < 1.2
# NSP > 4
# Fr > 500
