import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tools as Q

#lambdas
dFunc = {"time":lambda _df,col: _df[col],
         "speedup":lambda _df,col: _df['Extime']/_df[col]}


func_xscale = [lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('log', base=2),
              lambda _ax: _ax.set_xscale('linear')]

func_yscale = {"time": lambda _ax: _ax.set_yscale('log'),
               "speedup": lambda _ax: _ax.set_yscale('linear')}

func_ylim = {"time": lambda _ax,ymin,ymax: print(""),
             "speedup": lambda _ax,ymin,ymax: _ax.set_ylim([ymin, ymax])}

func_ylabel_lr = {"time": lambda _ax: _ax.yaxis.set_label_position("left"),
                  "speedup": lambda _ax: _ax.yaxis.set_label_position("right")}

func_adjust_margins = {"time": plt.subplots_adjust(),
                       "speedup": plt.subplots_adjust(bottom=0.1, left=0.02, right=0.93, top=0.99)}

func_ytick_lr = {"time": lambda _ax: _ax.yaxis.tick_left(),
                 "speedup": lambda _ax: _ax.yaxis.tick_right()}

dFixedOptimal = {"fixed":lambda n,g,r,B,_df,iVAR,ap,BSX,BSY: dfFixedFilter[iVAR](n,g,r,B,_df,BSX,BSY),
                 "optimal":lambda n,g,r,B,_df,iVAR,ap,BSX,BSY: dfOptimalFilter[iVAR](n,g,r,B,_df,ap,BSX,BSY)}

dfFixedFilter = [lambda n,g,r,B,_df,BSX,BSY: Q.fixedFilter(_df,'g','r','B',g,r,B,BSX,BSY),
                 lambda n,g,r,B,_df,BSX,BSY: Q.fixedFilter(_df,'n','r','B',n,r,B,BSX,BSY),
                 lambda n,g,r,B,_df,BSX,BSY: Q.fixedFilter(_df,'n','g','B',n,g,B,BSX,BSY),
                 lambda n,g,r,B,_df,BSX,BSY: Q.fixedFilter(_df,'n','g','r',n,g,r,BSX,BSY),
                 lambda n,g,r,B,_df,BSX,BSY: Q.fixedFilter_grB(_df,n,BSX,BSY)]

# optimal case
# - speedup vs {g,r,B} --> filter for the chosen n, then find the optimal g,r,B and explore B at those fixed {g,r} values.
# - speedup vs grB     --> filter for the chosen n, plot all and then highlight the optimal g,r,B point for each approach.
# - speedup vs n       --> filter nothing, then for each different 'n', find the optimal g,r,B of each approach.
dfOptimalFilter = [ lambda n,g,r,B,_df,ap,BSX,BSY: Q.optimalFilter_n(n,g,r,B,_df,ap+'time',BSX,BSY),
                    lambda n,g,r,B,_df,ap,BSX,BSY: Q.optimalFilter(n,g,r,B,_df,[0,1,1],ap+'time',BSX,BSY),
                    lambda n,g,r,B,_df,ap,BSX,BSY: Q.optimalFilter(n,g,r,B,_df,[1,0,1],ap+'time',BSX,BSY),
                    lambda n,g,r,B,_df,ap,BSX,BSY: Q.optimalFilter(n,g,r,B,_df,[1,1,0],ap+'time',BSX,BSY),
                    lambda n,g,r,B,_df,ap,BSX,BSY: Q.optimalFilter_grB(n,g,r,B,_df,BSX,BSY)]

dMeasure = {"time":0, "speedup":1}
dVAR = {"n":0, "g":1, "r":2, "B":3, "grB":4}
dVARStr = {"n":"n", "g":"g", "r":"r", "B":"B", "grB":"\{g,r,B\}"}
dLabel = {"time":"T", "speedup":r"S"}
dFuncText = [lambda text: text, lambda text: text, lambda text: text, lambda text: text, lambda text: ""]

dExText = {"time": lambda text: text, "speedup": lambda text: ""}


# ----------------
# 1) INIT code
# ----------------
if len(sys.argv)!=24:
    print("\nEjecutar como: python explot.py <datafile-prefix> <GPU_MODEL> <BSX0> <BSY0> <BSX1> <BSY1> <BSX2> <BSY2> <BSX3> <BSY3> <BSX4> <BSY4> <HWq> <HWc> <measure> <n> <g> <r> <B> <VAR> <OPT> <ymin> <ymax>")
    print("------------------------------------------------")
    print("datafile-prefix   : path+fileprefix before bsize being specified (check example below)")
    print("GPU_MODEL         : <string>")
    print("BSX{0,1,2,3,4}    : BSX config for Exhaustive, DP-SBR, DP-MBR, ASK-SBR and ASK-MBR respectively")
    print("BSY{0,1,2,3,4}    : BSY config for Exhaustive, DP-SBR, DP-MBR, ASK-SBR and ASK-MBR respectively")
    print("HWq               : [info] # of SMs in GPU")
    print("HWc               : [info] # of CUDA cores/SM")
    print("measure           : {time, speedup}")
    print("n                 : problem size n x n")
    print("g                 : initial subdivision of g x g)")
    print("r                 : subdivison of r x r")
    print("B                 : smallest regions of B x B")
    print("VAR               : {n, g, r, B, grB}")
    print("OPT               : {fixed, optimal} (fixed -> uses chosen {g,r,B})")
    print("{ymin,ymax}       : Y range for plots")
    print("------------------------------------------------")
    print("Example:\n python explot.py ../data/A100-ARCHsm_80 \"Nvidia A100\"   8 8   16 16   32 32   64 4   64 8   108 64 speedup   $((2**16)) 2 2 32  4 8 16 32 n 0 10\n\n")
    exit(2)

dfile_prefix = sys.argv[1]

GPUmodel = sys.argv[2]

BSX0 = int(sys.argv[3])
BSY0 = int(sys.argv[4])

BSX1 = int(sys.argv[5])
BSY1 = int(sys.argv[6])

BSX2 = int(sys.argv[7])
BSY2 = int(sys.argv[8])

BSX3 = int(sys.argv[9])
BSY3 = int(sys.argv[10])

BSX4 = int(sys.argv[11])
BSY4 = int(sys.argv[12])

q = int(sys.argv[13])
c = int(sys.argv[14])
measure = sys.argv[15]
n = int(sys.argv[16])
g = int(sys.argv[17])
r = int(sys.argv[18])
B = int(sys.argv[19])


VAR = sys.argv[20]
iVAR = dVAR[VAR]

OPT = sys.argv[21]

# y range
ymin = float(sys.argv[22])
ymax = float(sys.argv[23])

dxlabel = {"n":f"${dVARStr[VAR]}$", "g":f"${dVARStr[VAR]}$", "r":f"${dVARStr[VAR]}$", "B":f"${dVARStr[VAR]}$", "grB":f"Configuration Space ${dVARStr[VAR]}$"}
dTitle = {"time":f"Execution Time [s] vs ${dVARStr[VAR]}$", "speedup":f"Experimental Speedup $S({dVARStr[VAR]})$"}

fname_EX = os.path.basename(dfile_prefix+f"-BSX{BSX0}-BSY{BSY0}.dat")
fname_DP_SBR = os.path.basename(dfile_prefix+f"-BSX{BSX1}-BSY{BSY1}.dat")
fname_DP_MBR = os.path.basename(dfile_prefix+f"-BSX{BSX2}-BSY{BSY2}.dat")
fname_ASK_SBR = os.path.basename(dfile_prefix+f"-BSX{BSX3}-BSY{BSY3}.dat")
fname_ASK_MBR = os.path.basename(dfile_prefix+f"-BSX{BSX4}-BSY{BSY4}.dat")

print(f"[EXPLOT]> dfile_prefix='{dfile_prefix}'")
print(f"[EXPLOT]> GPU_MODEL={GPUmodel},  HWq={q}  HWc={c}\n")
print(f"[EXPLOT]> Exhaustive using '{fname_EX}'")
print(f"[EXPLOT]> DP_SBR using '{fname_DP_SBR}'")
print(f"[EXPLOT]> DP_MBR using '{fname_DP_MBR}'")
print(f"[EXPLOT]> ASK_SBR using '{fname_ASK_SBR}'")
print(f"[EXPLOT]> ASK_MBR using '{fname_ASK_MBR}'\n")

print(f"[EXPLOT]> measure={measure}, mode={OPT}")
print(f"[EXPLOT]> n={n} g={g} r={r} B={B}  VAR={VAR}")
print(f"[EXPLOT]> ymin={ymin} ymax={ymax}\n")











# -----------------------------------------------------------
# 2) load the data frame and filter according to parameters
# -----------------------------------------------------------
print(f"[EXPLOT]> Loading data frames...........", end='')
sys.stdout.flush()
df_EX       = pd.read_csv(dfile_prefix+f"-BSX{BSX0}-BSY{BSY0}.dat", comment='#')
df_DPSBR    = pd.read_csv(dfile_prefix+f"-BSX{BSX1}-BSY{BSY1}.dat", comment='#')
df_DPMBR    = pd.read_csv(dfile_prefix+f"-BSX{BSX2}-BSY{BSY2}.dat", comment='#')
df_ASKSBR   = pd.read_csv(dfile_prefix+f"-BSX{BSX3}-BSY{BSY3}.dat", comment='#')
df_ASKMBR   = pd.read_csv(dfile_prefix+f"-BSX{BSX4}-BSY{BSY4}.dat", comment='#')

#print(f"done: shape ->", df_EX.shape)
#print(f"done: shape ->", df_DPSBR.shape)
#print(f"done: shape ->", df_DPMBR.shape)
#print(f"done: shape ->", df_ASKSBR.shape)
#print(f"done: shape ->", df_ASKMBR.shape)
print("done")










# ----------------------------------------------------------
# 3) Filtering data frames
# ----------------------------------------------------------
print(f"[EXPLOT]> Filtering data frames.........", end='')
sys.stdout.flush()
df_EX, ftext_EX = dFixedOptimal[OPT](n,g,r,B,df_EX,iVAR,'Ex',BSX0,BSY0)
df_EX = Q.adapt_df(VAR, df_EX, df_EX)

df_DPSBR, ftext_DPSBR = dFixedOptimal[OPT](n,g,r,B,df_DPSBR,iVAR,'DPSBR',BSX1,BSY1)
df_DPSBR = Q.adapt_df(VAR, df_EX, df_DPSBR)

df_DPMBR, ftext_DPMBR = dFixedOptimal[OPT](n,g,r,B,df_DPMBR,iVAR,'DPMBR',BSX2,BSY2)
df_MBR = Q.adapt_df(VAR, df_EX, df_DPMBR)

df_ASKSBR, ftext_ASKSBR = dFixedOptimal[OPT](n,g,r,B,df_ASKSBR,iVAR,'ASKSBR',BSX3,BSY3)
df_ASKSBR = Q.adapt_df(VAR, df_EX, df_ASKSBR)

df_ASKMBR, ftext_ASKMBR = dFixedOptimal[OPT](n,g,r,B,df_ASKMBR,iVAR,'ASKMBR',BSX4,BSY4)
df_ASKMBR = Q.adapt_df(VAR, df_EX, df_ASKMBR)

#print(f"done: shape ->", df_EX.shape)
#print(f"done: shape ->", df_DPSBR.shape)
#print(f"done: shape ->", df_DPMBR.shape)
#print(f"done: shape ->", df_ASKSBR.shape)
#print(f"done: shape ->", df_ASKMBR.shape)
print("done")








# -----------------------------------------------------------
# 4) create 'measure' values for plotting
# -----------------------------------------------------------
# TODO
# GENERALIZAR ESTO (TIME Y SPEEDUP)
# aplicar scala logaritmica a eje x
print(f"[EXPLOT]> Creating measures.............", end='')
sys.stdout.flush()
EX_FUNC    = dFunc[measure](df_EX, 'Extime')
DPSBR_FUNC = dFunc[measure](df_DPSBR, 'DPSBRtime')
DPMBR_FUNC = dFunc[measure](df_DPMBR, 'DPMBRtime')
ASKSBR_FUNC = dFunc[measure](df_ASKSBR, 'ASKSBRtime')
ASKMBR_FUNC = dFunc[measure](df_ASKMBR, 'ASKMBRtime')
print("done")










# -----------------------------------------------------------
# 4) plot
# -----------------------------------------------------------
print(f"[EXPLOT]> Plotting......................", end='')
sys.stdout.flush()
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
#fig, ax = plt.subplots()
plt.suptitle(dTitle[measure], fontsize=12)
plt.title(Q.genSubtitleExp(GPUmodel, measure, VAR, n, g, r, B, q, c), fontsize=10, loc='center')

# curves
ax.plot(df_EX[VAR], EX_FUNC, Q.dStyle[iVAR][0], lw=1, markersize=4, label=dFuncText[iVAR](dExText[measure]('Exhaustive'+ftext_EX)),color=Q.cGrayscale[0], alpha=1.0)
ax.plot(df_DPSBR[VAR], DPSBR_FUNC, Q.dStyle[iVAR][3], lw=1, markersize=4, label=dFuncText[iVAR]('DP-SBR'+ftext_DPSBR),color=Q.cGreen[1],alpha=Q.alpha_grB[iVAR])
ax.plot(df_DPMBR[VAR], DPMBR_FUNC, Q.dStyle[iVAR][4], lw=1, markersize=4, label=dFuncText[iVAR]('DP-MBR'+ftext_DPMBR),color=Q.cPurple[0],alpha=Q.alpha_grB[iVAR])
ax.plot(df_ASKSBR[VAR], ASKSBR_FUNC, Q.dStyle[iVAR][1],lw=1,markersize=4,label=dFuncText[iVAR]('ASK-SBR'+ftext_ASKSBR),color=Q.cTemporal[0],alpha=Q.alpha_grB[iVAR])
ax.plot(df_ASKMBR[VAR], ASKMBR_FUNC, Q.dStyle[iVAR][2], lw=1, markersize=4,label=dFuncText[iVAR]('ASK-MBR'+ftext_ASKMBR),color=Q.cRed[0],alpha=Q.alpha_grB[iVAR])

# text on plot for grB landscape
Q.paintSpecialPoints(VAR, iVAR, ax, df_DPSBR, DPSBR_FUNC, BSX1, BSY1,
                                    df_DPMBR, DPMBR_FUNC, BSX2, BSY2,
                                    df_ASKSBR, ASKSBR_FUNC, BSX3, BSY3,
                                    df_ASKMBR, ASKMBR_FUNC, BSX4, BSY4);

# visual tweaks
ax.set_xlabel(fr'{dxlabel[VAR]}')
ax.set_ylabel(dLabel[measure], rotation=0, labelpad=7)
func_ylabel_lr[measure](ax)
func_ytick_lr[measure](ax)
func_xscale[iVAR](ax)
func_yscale[measure](ax)
func_ylim[measure](ax,ymin,ymax)
fig.tight_layout()
func_adjust_margins[measure]
plt.legend(prop={"size":10})
plt.savefig(f'../plots/exp-{GPUmodel}-{measure}-{VAR}-{OPT}.pdf', format='pdf')
plt.show()
print("done")
