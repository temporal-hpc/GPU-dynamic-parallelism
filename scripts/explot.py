import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tools as Q

#lambdas
dFunc = [  lambda _df: [_df['Extime'], _df['DPSBRtime'], _df['DPMBRtime'], _df['ASKSBRtime'], _df['ASKMBRtime']],
            lambda _df: [_df['Extime']/_df['Extime'], _df['Extime']/_df['DPSBRtime'], _df['Extime']/_df['DPMBRtime'], _df['Extime']/_df['ASKSBRtime'], _df['Extime']/_df['ASKMBRtime']]]


func_xscale = [lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('linear')]

dfFilter = [lambda n,g,r,B,_df: _df[                (_df['g']==g) & (_df['r']==r) & (_df['B']==B)],
            lambda n,g,r,B,_df: _df[(_df['n']==n)                 & (_df['r']==r) & (_df['B']==B)],
            lambda n,g,r,B,_df: _df[(_df['n']==n) & (_df['g']==g)                 & (_df['B']==B)],
            lambda n,g,r,B,_df: _df[(_df['n']==n) & (_df['g']==g) & (_df['r']==r)                ],
            lambda n,g,r,B,_df: _df[(_df['n']==n)                                                ]]

dStyle = [[':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v-','^-','+-','x-'],
          [':','v','^','+','x']]




# ----------------
# 1) INIT code
# ----------------
if len(sys.argv) !=13:
    print("\nEjecutar como: python explot.py <datafile> <GPU_MODEL> <HWq> <HWc> <measure> <n> <g> <r> <B> <VAR> <ymin> <ymax>")
    print("------------------------------------------------")
    print("datafile    : <path-to-file>")
    print("GPU_MODEL   : <string>")
    print("HWq         : # of SMs in GPU")
    print("HWc         : # of CUDA cores/SM")
    print("measure     : {time, speedup}")
    print("n           : problem size n x n")
    print("g           : initial subdivision of g x g)")
    print("r           : subdivison of r x r")
    print("B           : smallest regions of B x B")
    print("VAR         : {n, g, r, B, grB}")
    print("{ymin,ymax} : Y range for plots")
    print("------------------------------------------------")
    print("Example:\n python explot.py ../data/A100-ARCHsm_80-BSX64-BSY8.dat \"Nvidia A100\" 108 64 speedup   $((2**16)) 2 2 32  4 8 16 32   g n 0 10\n\n")
    exit(2)

datafile = sys.argv[1]
GPUmodel = sys.argv[2]
q = int(sys.argv[3])
c = int(sys.argv[4])
measure = sys.argv[5]
n = int(sys.argv[6])
g = int(sys.argv[7])
r = int(sys.argv[8])
B = int(sys.argv[9])


VAR = sys.argv[10]

# y range
ymin = float(sys.argv[11])
ymax = float(sys.argv[12])

# maps
dMeasure = {"time":0, "speedup":1}
dVAR = {"n":0, "g":1, "r":2, "B":3, "grB":4}
dVARStr = {"n":"n", "g":"g", "r":"r", "B":"B", "grB":"\{g,r,B\}"}
dxlabel = {"n":f"${dVARStr[VAR]}$", "g":f"${dVARStr[VAR]}$", "r":f"${dVARStr[VAR]}$", "B":f"${dVARStr[VAR]}$", "grB":f"Configuration Space ${dVARStr[VAR]}$"}
dLabel = {"time":"T", "speedup":r"S"}
dTitle = {"time":f"Execution Time [s] vs ${dVARStr[VAR]}$, {GPUmodel}", "speedup":f"Experimental Speedup $S({dVARStr[VAR]})$, {GPUmodel}"}

print(f"[EXPLOT]> GPU_MODEL={GPUmodel},  HWq={q}  HWc={c}")
print(f"[EXPLOT]> n={n} g={g} r={r} B={B}  VAR={VAR}")
print(f"[EXPLOT]> ymin={ymin} ymax={ymax}\n")











# -----------------------------------------------------------
# 2) load the data frame and filter according to parameters
# -----------------------------------------------------------
print(f"[EXPLOT]> Loading data frame...........", end='')
df = pd.read_csv(datafile, comment='#')
print(f"done: shape ->", df.shape)
subdf = dfFilter[dVAR[VAR]](n,g,r,B,df)
subdf = subdf.assign(grB=pd.Series(np.arange(len(subdf))).values)
print("[EXPLOT]> data frame:\n", subdf)











# -----------------------------------------------------------
# 3) create 'measure' values
# -----------------------------------------------------------
# TODO
# GENERALIZAR ESTO (TIME Y SPEEDUP)
# aplicar scala logaritmica a eje x
FUNCS = dFunc[dMeasure[measure]](subdf)
EX_FUNC     = FUNCS[0]
DPSBR_FUNC  = FUNCS[1]
DPMBR_FUNC  = FUNCS[2]
ASKSBR_FUNC = FUNCS[3]
ASKMBR_FUNC = FUNCS[4]


maxDPSBR = DPSBR_FUNC.max()
maxDPMBR = DPMBR_FUNC.max()
maxASKSBR = ASKSBR_FUNC.max()
maxASKMBR = ASKMBR_FUNC.max()

maxDPSBRi = DPSBR_FUNC.idxmax()
maxDPMBRi = DPMBR_FUNC.idxmax()
maxASKSBRi = ASKSBR_FUNC.idxmax()
maxASKMBRi = ASKMBR_FUNC.idxmax()

gDPSBR = subdf.at[maxDPSBRi, 'g']
rDPSBR = subdf.at[maxDPSBRi, 'r']
BDPSBR = subdf.at[maxDPSBRi, 'B']
grBDPSBR =subdf.at[maxDPSBRi, 'grB']


gDPMBR = subdf.at[maxDPMBRi, 'g']
rDPMBR = subdf.at[maxDPMBRi, 'r']
BDPMBR = subdf.at[maxDPMBRi, 'B']
grBDPMBR =subdf.at[maxDPMBRi, 'grB']

gASKSBR = subdf.at[maxASKSBRi, 'g']
rASKSBR = subdf.at[maxASKSBRi, 'r']
BASKSBR = subdf.at[maxASKSBRi, 'B']
grBASKSBR =subdf.at[maxASKSBRi, 'grB']

gASKMBR = subdf.at[maxASKMBRi, 'g']
rASKMBR = subdf.at[maxASKMBRi, 'r']
BASKMBR = subdf.at[maxASKMBRi, 'B']
grBASKMBR =subdf.at[maxASKMBRi, 'grB']

print(f"maxDPSBR -> x={grBDPSBR} S={maxDPSBR} ({gDPSBR},{rDPSBR},{BDPSBR})")
print(f"maxDPMBR -> x={grBDPMBR} S={maxDPMBR} ({gDPMBR},{rDPMBR},{BDPMBR})")
print(f"maxASKSBR -> x={grBASKSBR} S={maxASKSBR} ({gASKSBR},{rASKSBR},{BASKSBR})")
print(f"maxASKMBR -> x={grBASKMBR} S={maxASKMBR} ({gASKMBR},{rASKMBR},{BASKMBR})")









# -----------------------------------------------------------
# 4) plot
# -----------------------------------------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
#fig, ax = plt.subplots()
plt.suptitle(dTitle[measure], y=0.98, fontsize=12)
plt.title(Q.genSubtitleExp(measure,VAR,n,g,r,B,q,c), fontsize=10)

# curves
ax.plot(subdf[VAR], EX_FUNC, dStyle[dVAR[VAR]][0], lw=1, markersize=4, color=Q.cGrayscale[0])
ax.plot(subdf[VAR], ASKSBR_FUNC, dStyle[dVAR[VAR]][1], lw=1, markersize=3, color=Q.cTemporal[1], alpha=Q.alpha_grB)
ax.plot(subdf[VAR], ASKMBR_FUNC, dStyle[dVAR[VAR]][2], lw=1, markersize=3, color=Q.cRed[0], alpha=Q.alpha_grB)
ax.plot(subdf[VAR], DPMBR_FUNC, dStyle[dVAR[VAR]][3], lw=1, markersize=4,  color=Q.cGreen[0], alpha=Q.alpha_grB)
ax.plot(subdf[VAR], DPSBR_FUNC, dStyle[dVAR[VAR]][4], lw=1, markersize=4,  color=Q.cPurple[2], alpha=Q.alpha_grB)

# text on plot for grB landscape
if(VAR=='grB'):
    plt.plot(grBASKSBR, maxASKSBR, dStyle[dVAR[VAR]][1], markersize=4,  label="ASK-SBR", color=Q.cTemporal[1])
    plt.plot(grBASKMBR, maxASKMBR, dStyle[dVAR[VAR]][2], markersize=4,  label="ASK-MBR", color=Q.cRed[0])
    plt.plot(grBDPSBR, maxDPSBR, dStyle[dVAR[VAR]][3], markersize=7,    label="DP-SBR",  color=Q.cGreen[0])
    plt.plot(grBDPMBR, maxDPMBR, dStyle[dVAR[VAR]][4], markersize=6,    label="DP-MBR",  color=Q.cPurple[2])

    plt.text(grBDPSBR+5, maxDPSBR, f"({gDPSBR},{rDPSBR},{BDPSBR})",      fontsize=8, fontweight='bold')
    plt.text(grBDPMBR+5, maxDPMBR, f"({gDPMBR},{rDPMBR},{BDPMBR})",      fontsize=8, fontweight='bold')
    plt.text(grBASKSBR+5, maxASKSBR, f"({gASKSBR},{rASKSBR},{BASKSBR})", fontsize=8, fontweight='bold')
    plt.text(grBASKMBR+5, maxASKMBR, f"({gASKMBR},{rASKMBR},{BASKMBR})", fontsize=8, fontweight='bold')

ax.set_xlabel(fr'{dxlabel[VAR]}')
ax.set_ylabel(dLabel[measure], rotation=0, labelpad=5)
func_xscale[dVAR[VAR]](ax)
ax.set_ylim([ymin, ymax])

if(VAR=='grB'):
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    ax.set_xticklabels([])

#fig.tight_layout()
plt.subplots_adjust(bottom=0.05, left=0.06, right=0.98, top=0.90)
plt.legend(prop={"size":10})
plt.savefig(f'plots/explot-{GPUmodel}-q{q}-c{c}-{measure}-{VAR}.eps', format='eps')
plt.show()

print("END\n\n")
