import pandas as pd
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

alpha_grB=[1.0, 1.0, 1.0, 1.0, 0.2]

dStyle = [[':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v','^','+','x']]

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

def genSubtitleExp(GPUmodel, measure,VAR, n, g, r, B, q, c):
    subtitle = f"{GPUmodel}, "
    #if VAR != "g" and VAR != "grB":
    #    subtitle += f'$g={g}$, '
    #if VAR != "r" and VAR != "grB":
    #    subtitle += f'$r={r}$, '
    #if VAR != "B" and VAR != "grB":
    #    subtitle += f'$B={B}$, '
    subtitle += f'$q={q}, c={c}$'
    if VAR != "n":
        subtitle += r", $n=2^{" + f"{int(np.log2(n))}" + r"}$"

    return subtitle

def genLabel(MVAR, mvarSTR, val):
    if MVAR == "n":
        exp = int(np.log2(val))
        return f"${mvarSTR} = "+ "2^{" + f"{exp}" + "}$"
    else:
        return f"${mvarSTR} = "+"{:g}".format(val)+"$"

# fixed filter for n,g,r, or B
def fixedFilter(_df,p1,p2,p3, v1,v2,v3, BSX, BSY):
    # filter for the chosen n
    subdf = _df[(_df[p1]==v1) & (_df[p2]==v2) & (_df[p3]==v3)]
    subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
    ftext = fr"@${BSX} \times {BSY},{p1}={v1},{p2}={v2},{p3}={v3}$"
    return subdf,ftext

# fixed filter for grB landscape
def fixedFilter_grB(_df,n, BSX, BSY):
    # filter for the chosen n
    subdf = _df[(_df['n']==n)]
    subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
    ftext = fr"@${BSX}\times{BSY}$"
    return subdf,ftext

# optimal filter when exploring n
def optimalFilter_n(n,g,r,B,_df,col, BSX, BSY):
    nvalues = _df['n'].unique()
    subdf = pd.DataFrame()
    for i in nvalues:
        #print(f"{i}")
        ndf = _df[(_df['n']==i)]
        bestIndex = ndf[col].idxmin()
        g = ndf.at[bestIndex, 'g']
        r = ndf.at[bestIndex, 'r']
        B = ndf.at[bestIndex, 'B']
        #print(f"bestIndex = {bestIndex}, g={g}, r={r}, B={B}")
        ndf = ndf[(ndf['g']==g) & (ndf['r']==r) & (ndf['B']==B)]
        subdf = subdf.append(ndf)
        subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
        #print(ndf)
    #print(subdf)
    #exit(1)
    ftext = fr"@${BSX}\times{BSY}$"
    return subdf,ftext

# optimal filter for g,r or B
def optimalFilter(n,g,r,B,_df,param1,param2,col, BSX, BSY):
    # filter for the chosen n
    subdf = _df[(_df['n']==n)]
    # find the optimal r,g,B tuple
    bestIndex = subdf[col].idxmin()
    val1 = subdf.at[bestIndex, param1]
    val2 = subdf.at[bestIndex, param2]
    # filter according to var1 and var2
    subdf = subdf[(subdf[param1]==val1) & (subdf[param2]==val2)]
    subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
    #print(f"\nOptimal params for {col}:  {param1}={val1} {param2}={val2}")
    #print(f"{col} dataframe:\n",subdf)
    if col=="Extime":
        ftext = fr"@$BS={BSX}\times{BSY}$"
    else:
        ftext = fr"@$BS={BSX}\times{BSY},{param1}={val1},{param2}={val2}$"
    return subdf,ftext

# optimal filter when exploring grB lanscape
def optimalFilter_grB(n,g,r,B,_df, BSX, BSY):
    subdf = _df[(_df['n']==n)]
    subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
    ftext = fr"@${BSX}\times{BSY}$"
    return subdf,ftext

def paintSpecialPoints(VAR, iVAR, ax,
                       df_DPSBR, DPSBR_FUNC, BSX1, BSY1,
                       df_DPMBR, DPMBR_FUNC, BSX2, BSY2,
                       df_ASKSBR, ASKSBR_FUNC, BSX3, BSY3,
                       df_ASKMBR, ASKMBR_FUNC, BSX4, BSY4):
    if(VAR!='grB'):
        return

    # maximum tuples
    #maxDPSBR = DPSBR_FUNC.max()
    #maxDPMBR = DPMBR_FUNC.max()
    #maxASKSBR = ASKSBR_FUNC.max()
    #maxASKMBR = ASKMBR_FUNC.max()

    #maxDPSBRi = DPSBR_FUNC.idxmax()
    #maxDPMBRi = DPMBR_FUNC.idxmax()
    #maxASKSBRi = ASKSBR_FUNC.idxmax()
    #maxASKMBRi = ASKMBR_FUNC.idxmax()

    minDPSBR = df_DPSBR['DPSBRtime'].min()
    minDPMBR = df_DPSBR['DPMBRtime'].min()
    minASKSBR = df_DPSBR['ASKSBRtime'].min()
    minASKMBR = df_DPSBR['ASKMBRtime'].min()

    minDPSBRi = df_DPSBR['DPSBRtime'].idxmin()
    minDPMBRi = df_DPSBR['DPMBRtime'].idxmin()
    minASKSBRi = df_DPSBR['ASKSBRtime'].idxmin()
    minASKMBRi = df_DPSBR['ASKMBRtime'].idxmin()

    optDPSBR = DPSBR_FUNC.at[minDPSBRi]
    optDPMBR = DPMBR_FUNC.at[minDPMBRi]
    optASKSBR = ASKSBR_FUNC.at[minASKSBRi]
    optASKMBR = ASKMBR_FUNC.at[minASKMBRi]

    gDPSBR = df_DPSBR.at[minDPSBRi, 'g']
    rDPSBR = df_DPSBR.at[minDPSBRi, 'r']
    BDPSBR = df_DPSBR.at[minDPSBRi, 'B']
    grBDPSBR =df_DPSBR.at[minDPSBRi, 'grB']

    gDPMBR = df_DPMBR.at[minDPMBRi, 'g']
    rDPMBR = df_DPMBR.at[minDPMBRi, 'r']
    BDPMBR = df_DPMBR.at[minDPMBRi, 'B']
    grBDPMBR =df_DPMBR.at[minDPMBRi, 'grB']

    gASKSBR = df_ASKSBR.at[minASKSBRi, 'g']
    rASKSBR = df_ASKSBR.at[minASKSBRi, 'r']
    BASKSBR = df_ASKSBR.at[minASKSBRi, 'B']
    grBASKSBR =df_ASKSBR.at[minASKSBRi, 'grB']

    gASKMBR = df_ASKMBR.at[minASKMBRi, 'g']
    rASKMBR = df_ASKMBR.at[minASKMBRi, 'r']
    BASKMBR = df_ASKMBR.at[minASKMBRi, 'B']
    grBASKMBR =df_ASKMBR.at[minASKMBRi, 'grB']

    #print(f"maxDPSBR -> x={grBDPSBR} S={maxDPSBR} ({gDPSBR},{rDPSBR},{BDPSBR})")
    #print(f"maxDPMBR -> x={grBDPMBR} S={maxDPMBR} ({gDPMBR},{rDPMBR},{BDPMBR})")
    #print(f"maxASKSBR -> x={grBASKSBR} S={maxASKSBR} ({gASKSBR},{rASKSBR},{BASKSBR})")
    #print(f"maxASKMBR -> x={grBASKMBR} S={maxASKMBR} ({gASKMBR},{rASKMBR},{BASKMBR})")

    plt.plot(grBASKSBR, optASKSBR, dStyle[iVAR][1], markersize=5,  label=fr"ASK-SBR@$BS={BSX3}\times{BSY3}$", color=cTemporal[1])
    plt.plot(grBASKMBR, optASKMBR, dStyle[iVAR][2], markersize=5,  label=fr"ASK-MBR@$BS={BSX4}\times{BSY4}$", color=cRed[0])
    plt.plot(grBDPSBR, optDPSBR, dStyle[iVAR][3], markersize=8,    label=fr"DP-SBR@$BS={BSX1}\times{BSY1}$",  color=cGreen[0])
    plt.plot(grBDPMBR, optDPMBR, dStyle[iVAR][4], markersize=6,    label=fr"DP-MBR@$BS={BSX2}\times{BSY2}$",  color=cPurple[2])

    plt.text(grBDPSBR+5, optDPSBR, f"({gDPSBR},{rDPSBR},{BDPSBR})",      fontsize=8, fontweight='bold')
    plt.text(grBDPMBR+5, optDPMBR, f"({gDPMBR},{rDPMBR},{BDPMBR})",      fontsize=8, fontweight='bold')
    plt.text(grBASKSBR+5, optASKSBR, f"({gASKSBR},{rASKSBR},{BASKSBR})", fontsize=8, fontweight='bold')
    plt.text(grBASKMBR+5, optASKMBR, f"({gASKMBR},{rASKMBR},{BASKMBR})", fontsize=8, fontweight='bold')

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    ax.set_xticklabels([])

def adapt_df(VAR, dfREF, df):
    df.drop('Extime', 1)
    if VAR=='n' or VAR=='grB':
        ExVals = dfREF['Extime'].values
        #print("df ref\n",dfREF)
        #print("df before\n",df)
        df = df.assign(Extime=ExVals)
        #print("df after\n",df)
    else:
        ExMin = dfREF['Extime'].min()
        ExVals = np.full(len(df), ExMin)
        df = df.assign(Extime=ExVals)
    return df
