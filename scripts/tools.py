import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# list of color gradients (from bold to light)
colorA = ["#e66101", "#e07123", "#d6874a", "#c99d78"]
colorB = ["#5e3c99", "#755b99", "#857498", "#959496"]
colorC = ["#58957e", "#0099FF"]

cMapPurple = ["#8da0cb", "#97a8d2", "#a4b4db", "#b8c5ea"]
cMapGreen = ["#66c2a5", "#6ec5aa", "#7dccb4", "#8ad2bc"]
cMapOrange = ["#fc8d62", "#fb9871", "#fa9f7b", "#f9a685"]

cTemporal = ["#006e76", "#32868d", "#409097", "#519ea4"]
cOrange = ["#D8693E", "#d87751", "#d8815f", "#d59880"]
cRed = ["#cb181d", "#fb6a4a", "#fcae91", "#fee5d9"]
cGreen = ["#238b45", "#74c476", "#bae4b3", "#edf8e9"]
cPurple = ["#8860b4", "#9473b8", "#9d81bc", "#ab99c0"]
cGrayscale = ["#111111", "#333333", "#888888", "#CCCCCC"]

alpha_grB=0.2


# list of colors for classB

# The parametrized function to be plotted
def exhaustive(n, A, q, c):
    return A * np.ceil((n**2.0)/(q*c))


# Work subdiv [actualizado r-> lineal]
# single block per region (SBR)
def subdivSBR(n, g, B, r, P, lam, A, q, c):
    R = r*r
    G = g*g
    tau = np.floor(np.log(n/(g*B)) / np.log(r))
    result = np.zeros(len(n))
    for k in range(0, len(n)):
        sum = 0
        for i in range(0, int(tau[k])-1):
            Q = np.ceil(A[k] * (4.0*n[k]/(g[k]*(r[k]**(i))*c[k])))
            T = np.ceil((n[k]**2.0) / (G[k]*(R[k]**(i))*c[k]))
            sum += (Q + P[k]*lam[k]*A[k] + (1-P[k])*T) * (P[k]**i) * np.ceil(G[k]*(R[k]**(i))/q[k])
        K = sum
        #print(f"n[{k}]={n[k]} g[{k}]={g[k]} B[{k}]={B[k]} r[{k}]={r[k]} P[{k}]={P[k]} lam[{k}]={lam[k]} A[{k}]={A[k]} q[{k}]={q[k]} c[{k}]={c[k]}  tau[{k}]={tau[k]}")
        #print(f"K = {K}")
# ceil version 1.0 (outdated) # L = A[k] * (P[k]**(tau[k]-1)) * np.ceil((n[k]**2.0)/(q[k]*c[k])) # ceil version 2.0
        L = A[k] *  np.ceil( (n[k]**2.0)/(G[k]*(R[k]**(tau[k]-1))*c[k]) ) * np.ceil( (G[k]*(R[k]**(tau[k]-1)))/q[k] ) * (P[k]**(tau[k]-1))
        #print(f"L = {L}")
        result[k] = K + L
        #print(f"W_S = {K + L}\n")

        # Exhaustive part
        #E = np.ceil((n[k]**2.0)/(q[k]*c[k]))*A[k]
        #print(f"W_E = {E}\n")
        #print(f"E/Sub = {E/result[k]}\n")
        #input("Press Enter to continue...")
    return result


# multiple blocks per region (MBR)
def subdivMBR(n, g, B, r, P, lam, A, q, c):
    R = r*r
    G = g*g
    tau = np.floor(np.log(n/(g*B)) / np.log(r))
    result = np.zeros(len(n))
    for k in range(0, len(n)):
        sum = 0
        H = np.ceil( (n[k]**2.0)/(q[k] * c[k]))
        CS = lam[k]*A[k]
        for i in range(0, int(tau[k])-1):
            M1 = np.ceil((4.0*n[k]*A[k])/(g[k] * (r[k]**i) * c[k])) * np.ceil( (G[k]*(R[k]**i))/q[k] ) * P[k]**i
            M2 = np.ceil((G[k]*R[k]**i)/(q[k])) * CS * (P[k]**(i+1))
            M3 = H * (P[k]**i) * (1-P[k])
            sum += (M1 + M2 + M3)
        K = sum
        #print(f"n[{k}]={n[k]} g[{k}]={g[k]} B[{k}]={B[k]} r[{k}]={r[k]} P[{k}]={P[k]} lam[{k}]={lam[k]} A[{k}]={A[k]} q[{k}]={q[k]} c[{k}]={c[k]}  tau[{k}]={tau[k]}")
        #print(f"K = {K}")

        # new ceil version
        L = A[k] * (P[k]**(tau[k]-1)) * H

        #print(f"L = {L}")
        result[k] = K + L
        #print(f"W_S = {K + L}\n")

        # Exhaustive part
        #E = np.ceil((n[k]**2.0)/(q[k]*c[k]))*A[k]
        #print(f"W_E = {E}\n")
        #print(f"E/Sub = {E/result[k]}\n")
        #input("Press Enter to continue...")
    return result


def exhaustiveWork(n, A):
    o = np.full(len(n), int(1))
    return exhaustive(n, A, o, o)


def subdivWork(n, g, B, r, P, lam, A):
    o = np.full(len(n), int(1))
    return subdivSBR(n, g, B, r, P, lam, A, o, o)


def minmaxFromLists(m, d1, d2, d3, d4, d5):
    return m([d1.min(),d2.min(),d3.min(),d4.min(),d5.min()])


def genSubtitle(measure,MVAR,VAR, n, g, B, r, P, lam, A, q, c):
    subtitle = ""
    if VAR != "n" and MVAR != "n":
        subtitle += r"$n=2^{" + f"{int(np.log2(n[0]))}" + "}$, "
    if VAR != "g" and MVAR != "g":
        subtitle += f'g={g[0]}, '
    if VAR != "B" and MVAR != "B":
        subtitle += f'B={B[0]}, '
    if VAR != "P" and MVAR != "P":
        subtitle += f'P={P[0]}, '
    if VAR != "r" and MVAR != "r":
        subtitle += f'r={r[0]}, '
    if VAR != "lam" and MVAR != "lam":
        subtitle += f'$\lambda$={lam[0]}, '
    if VAR != "A" and MVAR != "A":
        subtitle += f'$C_A$={A[0]}'

    if measure == "speedup" or measure == "speedup-sbr" or measure == "speedup-mbr":
        if VAR != "q" and MVAR != "q":
            subtitle += f', $q$={q[0]}'
        if VAR != "c" and MVAR != "c":
            subtitle += f', $c$={c[0]}'

    return subtitle

def genSubtitleExp(measure,VAR, n, g, r, B, q, c):
    subtitle = ""
    if VAR != "n":
        subtitle += r"$n=2^{" + f"{int(np.log2(n))}" + r"}$, "
    if VAR != "g" and VAR != "grB":
        subtitle += f'$g={g}$, '
    if VAR != "r" and VAR != "grB":
        subtitle += f'$r={r}$, '
    if VAR != "B" and VAR != "grB":
        subtitle += f'$B={B}$, '

    if measure == "speedup":
        subtitle += f'$q={q}, c={c}$'

    return subtitle

def genLabel(MVAR, mvarSTR, val):
    if MVAR == "n":
        exp = int(np.log2(val))
        return f"${mvarSTR} = "+ "2^{" + f"{exp}" + "}$"
    else:
        return f"${mvarSTR} = "+"{:g}".format(val)+"$"
