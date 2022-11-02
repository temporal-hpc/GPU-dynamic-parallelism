import numpy as np
import matplotlib.pyplot as plt
import sys
import tools as Q


# functional mapping for [work, wrf, time, speedup, speedup-sbr, speedup-mbr]
funcs = [lambda n,g,B,r,P,lam,A,q,c: Q.subdivWork(n, g, B, r, P, lam, A),
         lambda n,g,B,r,P,lam,A,q,c: Q.exhaustiveWork(n,A)/Q.subdivWork(n, g, B, r, P, lam, A),
         lambda n,g,B,r,P,lam,A,q,c: Q.subdivSBR(n, g, B, r, P, lam, A, q, c),
         lambda n,g,B,r,P,lam,A,q,c: [Q.exhaustive(n,A,q,c)/Q.subdivSBR(n, g, B, r, P, lam, A, q, c),
                                        Q.exhaustive(n,A,q,c)/Q.subdivMBR(n, g, B, r, P, lam, A, q, c)],
         lambda n,g,B,r,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.subdivSBR(n, g, B, r, P, lam, A, q, c),
         lambda n,g,B,r,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.subdivMBR(n, g, B, r, P, lam, A, q, c)]


# functional mapping for [work, wrf, time, speedup-sbr, speedup-mbr]
refFuncs = [lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustiveWork(n,A),
            lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustiveWork(n,A)/Q.exhaustiveWork(n,A),
            lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustive(n,A,q,c),
            lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.exhaustive(n,A,q,c),
            lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.exhaustive(n,A,q,c)]


# functional mapping for [work, wrf, time, speedup-sbr, speedup-mbr]
setYscale = [lambda ax: ax.set_yscale('log', base=2),
             lambda ax: ax.set_yscale('linear'),
             lambda ax: ax.set_yscale('log', base=2),
             lambda ax: ax.set_yscale('linear'),
             lambda ax: ax.set_yscale('linear')]


# functional mapping for [work, wrf, time, speedup-sbr, speedup-mbr]
setYlim = [lambda ax,ymin,ymax,dmin,dmax: ax.set_ylim(dmin, 1.2*dmax),
           lambda ax,ymin,ymax,dmin,dmax: ax.set_ylim(ymin, ymax),
           lambda ax,ymin,ymax,dmin,dmax: ax.set_ylim(dmin, 1.2*dmax),
           lambda ax,ymin,ymax,dmin,dmax: ax.set_ylim(ymin, ymax),
           lambda ax,ymin,ymax,dmin,dmax: ax.set_ylim(ymin, ymax)]


# functional mapping of variables
mvarMap = [lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [mvar, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P,    lam,  A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , mvar, BSBR, rSBR, mvar, BMBR, rMBR, P,    lam,  A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, mvar, rSBR, gMBR, mvar, rMBR, P,    lam,  A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, mvar, gMBR, BMBR, mvar, P,    lam,  A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, mvar, lam,  A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P,    mvar, A,    q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P,    lam,  mvar, q,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P,    lam,  A, mvar,    c],
           lambda mvar,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [n   , gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P,    lam,  A,    q, mvar]]


# functional mapping for [work, wrf, time, speedup, speedup-sbr, speedup-mbr]
funcs = [lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.subdivWork(n, gSBR, BSBR, rSBR, P, lam, A),
         lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustiveWork(n,A)/Q.subdivWork(n, gSBR, BSBR, rSBR, P, lam, A),
         lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.subdivSBR(n, gSBR, BSBR, rSBR, P, lam, A, q, c),
         lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: [Q.exhaustive(n,A,q,c)/Q.subdivSBR(n, gSBR, BSBR, rSBR, P, lam, A, q, c),
                                        Q.exhaustive(n,A,q,c)/Q.subdivMBR(n, gMBR, BMBR, rMBR, P, lam, A, q, c)],
         lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.subdivSBR(n, gSBR, BSBR, rSBR, P, lam, A, q, c),
         lambda n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c: Q.exhaustive(n,A,q,c)/Q.subdivMBR(n, gMBR, BMBR, rMBR, P, lam, A, q, c)]



def genplot(measure,VAR,MVAR, mvarx, n, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P, lam, A, q, c, mode):
    if(mode=="optimal"):
        #print(f"Using optimal grB...")
        vRange = 2 ** np.arange(1,9)
        v = mvarMap[dmvar[MVAR]](mvarx,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c)
        #print("OPTIMAL SBR")
        #opt_gSBR, opt_rSBR, opt_BSBR = Q.opt_grB_range(measure,VAR, v[0], v[1], v[2], v[3], v[7], v[8], v[9], v[10], v[11], vRange, Q.subdivSBR_scalar)
        opt_gSBR, opt_rSBR, opt_BSBR = Q.opt_grB_range(measure,VAR, v[0], v[1], v[2], v[3], constP, constlam, constA, constq, constc, vRange, Q.subdivSBR_scalar)
        #print("OPTIMAL MBR")
        opt_gMBR, opt_rMBR, opt_BMBR = Q.opt_grB_range(measure,VAR, v[0], v[4], v[5], v[6], constP, constlam, constA, constq, constc, vRange, Q.subdivMBR_scalar)
        v[1] = opt_gSBR
        v[2] = opt_BSBR
        v[3] = opt_rSBR
        v[4] = opt_gMBR
        v[5] = opt_BMBR
        v[6] = opt_rMBR
        if VAR=="g":
            v[1] = gSBR
            v[4] = gMBR
        if VAR=="B":
            v[2] = BSBR
            v[5] = BMBR
        if VAR=="r":
            v[3] = rSBR
            v[6] = rMBR
        #print(f"v1={v[1]}")
        #print(f"v2={v[2]}")
        #print(f"v3={v[3]}")
        v = mvarMap[dmvar[MVAR]](mvarx,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11])
        d = funcs[dFunc[measure]](v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11])
        #print(f"genplot-optimal v={v}")
        #print(f"genplot-optimal d={d}")
        #print(f"done\n")
        return v,d
    else:
        v = mvarMap[dmvar[MVAR]](mvarx,n,gSBR,BSBR,rSBR,gMBR,BMBR,rMBR,P,lam,A,q,c)
        d = funcs[dFunc[measure]](v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11])
        #print(f"genplot v={v}")
        #print(f"genplot d={d}")
        return v,d



# ---------------------------------------------------
# main code
# ---------------------------------------------------
if len(sys.argv) !=23:
    print("\nEjecutar como python <prog> <measure> <n> <g> <B> <r> <P> <lam> <A> <q> <c> <mvar1> <mvar2> <mvar3> <mvar4> <MVAR> <VAR> <xmin> <xmax> <ymin> <ymax> <res> <mode>")
    print("measure = {work, wrf, time, speedup, speedup-sbr, speedup-mbr}")
    print("MVAR = {n, g, B, r, P, lam, A, q, c}")
    print("VAR = {n, g, B, r, P, lam, A, q, c}")
    print("res = number of points in [xmin, xmax]")
    print("mode = {fixed -> given {g,r,B}, optimal -> best {r,g,B}}")
    exit(2)

# parameters
res = int(sys.argv[21])
measure = sys.argv[1]
n = np.full(res, int(sys.argv[2]))
gSBR = np.full(res, int(sys.argv[3]))
BSBR = np.full(res, int(sys.argv[4]))
rSBR = np.full(res, int(sys.argv[5]))
gMBR = np.full(res, int(sys.argv[3]))
BMBR = np.full(res, int(sys.argv[4]))
rMBR = np.full(res, int(sys.argv[5]))
P = np.full(res, float(sys.argv[6]))
lam = np.full(res, float(sys.argv[7]))
A = np.full(res, float(sys.argv[8]))
q = np.full(res, int(sys.argv[9]))
c = np.full(res, int(sys.argv[10]))

constP = P
constlam = lam
constA = A
constq = q
constc = c


# multi parameters
mvar1 = np.full(res, float(sys.argv[11]))
mvar2 = np.full(res, float(sys.argv[12]))
mvar3 = np.full(res, float(sys.argv[13]))
mvar4 = np.full(res, float(sys.argv[14]))

MVAR = sys.argv[15]
VAR = sys.argv[16]
mode = sys.argv[22]

# maps
dFunc = {"work":0, "wrf":1, "time":2, "speedup":3, "speedup-sbr":4, "speedup-mbr":5}
dmvar = {"n":0, "g":1, "B":2, "r":3, "P":4, "lam":5, "A":6, "q":7, "c":8}
dmvarStr = {"n":"n", "g":"g", "B":"B", "r":"r", "P":"P", "lam":"\lambda", "A":"\mathcal{A}", "q":"q", "c":"c"}
dLabel = {"work":r"${W}$",
          "wrf":r"$\Omega$",
          "time":"T",
          "speedup":r"S",
          "speedup-sbr":r"$S_{\textit{SBR}}$",
          "speedup-mbr":r"$S_{\textit{MBR}}$"}
dTitle = {"work":f"Theoretical Work vs ${dmvarStr[VAR]}$",
          "wrf":f"Theoretical $\Omega({VAR})$, multi-${dmvarStr[MVAR]}$",
          "time":f"Theoretical Execution Time [s] vs ${dmvarStr[VAR]}$",
          "speedup":f"Theoretical $S({VAR})$, multi-${dmvarStr[MVAR]}$",
          "speedup-sbr":r"Theoretical $S_{\textit{SBR}}$"+f"$({VAR})$, multi-${dmvarStr[MVAR]}$",
          "speedup-mbr":r"Theoretical $S_{\textit{MBR}}$"+f"$({VAR})$, multi-${dmvarStr[MVAR]}$"}
# x range
xmin = float(sys.argv[17])
xmax = float(sys.argv[18])

# y range
ymin = float(sys.argv[19])
ymax = float(sys.argv[20])

# creating xrange
print(f"[CMODEL]> MVAR={MVAR} VAR={VAR}")
print(f"[CMODEL]> n={n[0]}")
print(f"[CMODEL]> gSBR={gSBR[0]} BSBR={BSBR[0]} rSBR={rSBR[0]}")
print(f"[CMODEL]> gMBR={gMBR[0]} BMBR={BMBR[0]} rMBR={rMBR[0]}")
print(f"[CMODEL]> P={P[0]} lam={lam[0]} A={A[0]} q={q[0]} c={c[0]}")
print(f"[CMODEL]> xmin={xmin} xmax={xmax} ymin={ymin} ymax={ymax} res={res}")
print(f"[CMODEL]> dFunc[{measure}] = {dFunc[measure]}")
print(f"[CMODEL]> # of points = {res}")
xrange = np.linspace(xmin, xmax, res)
# creating plot
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot()
#fig, ax = plt.subplots()
plt.suptitle(dTitle[measure], y=0.93, fontsize=10)

ax.set_xlabel(fr'${dmvarStr[VAR]}$')
ax.set_ylabel(dLabel[measure], rotation=0, labelpad=15)

if VAR == "n":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    n = xrange
    ax.set_xscale('log', base=2)
if VAR == "g":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    gSBR = xrange
    gMBR = xrange
    ax.set_xscale('log', base=2)
if VAR == "B":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    #xrange = vRange = 2 ** np.arange(1,10)
    BSBR = xrange
    BMBR = xrange
    ax.set_xscale('log', base=2)
    #print(f"xrange for B = {xrange}")
if VAR == "P":
    P = xrange
if VAR == "r":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    rSBR = xrange
    rMBR = xrange
    ax.set_xscale('log', base=2)
if VAR == "lam":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=10)
    lam = xrange
    ax.set_xscale('log', base=10)
if VAR == "A":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    lam = xrange
    ax.set_xscale('log', base=2)
if VAR == "q":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    q = xrange
    ax.set_xscale('log', base=2)
if VAR == "c":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    c = xrange
    ax.set_xscale('log', base=2)



#if measure == "speedup" or measure == "speedup-sbr" or measure == "speedup-mbr":
#    plt.title(Q.genSubtitle(measure,MVAR,VAR,n,gSBR,BSBR,rSBR,P,lam,A,q,c), fontsize=8)
#else:
plt.title(Q.genSubtitle(measure,MVAR,VAR,n,gSBR,BSBR,rSBR,P,lam,A,q,c), fontsize=9)


v1,d1 = genplot(measure,VAR,MVAR,mvar1, n, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P, lam, A, q, c, mode)
v2,d2 = genplot(measure,VAR,MVAR,mvar2, n, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P, lam, A, q, c, mode)
v3,d3 = genplot(measure,VAR,MVAR,mvar3, n, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P, lam, A, q, c, mode)
v4,d4 = genplot(measure,VAR,MVAR,mvar4, n, gSBR, BSBR, rSBR, gMBR, BMBR, rMBR, P, lam, A, q, c, mode)
d5 = refFuncs[dFunc[measure]](v1[0],v1[1],v1[2],v1[3],v1[4],v1[5],v1[6],v1[7],v1[8],v1[9],v1[10],v1[11])

# refplot
if measure == "time" or measure == "work":
    refPlot,     = plt.plot(xrange, d5, label="Ex, "+Q.genLabelRef(MVAR, dmvarStr[MVAR], mvar4[0]), lw=2, ls=':', color=Q.cGrayscale[2])
else:
    refPlot,     = plt.plot(xrange, d5, lw=0.5, ls='--', color=Q.cGrayscale[2])

# mainplot
if measure=="speedup":
    # SBR part
    subdivPlot1, = plt.plot(xrange, d1[0], label="SBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_1', mvar1[0], v1[1], v1[2], v1[3]), lw=1, ls=":", color=Q.cTemporal[3], alpha=0.7)
    subdivPlot2, = plt.plot(xrange, d2[0], label="SBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_2', mvar2[0], v2[1], v2[2], v2[3]), lw=1, ls="-.", color=Q.cTemporal[2], alpha=0.7)
    subdivPlot3, = plt.plot(xrange, d3[0], label="SBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_3', mvar3[0], v3[1], v3[2], v3[3]), lw=1, ls="--", color=Q.cTemporal[1], alpha=0.7)
    subdivPlot4, = plt.plot(xrange, d4[0], label="SBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_4', mvar4[0], v4[1], v4[2], v4[3]), lw=1, ls="-", color=Q.cTemporal[0], alpha=0.7)

    # MBR part
    subdivPlot1b, = plt.plot(xrange, d1[1], label="MBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_1', mvar1[0], v1[4], v1[5], v1[6]), lw=1, ls=":", color=Q.cOrange[3], alpha=0.7)
    subdivPlot2b, = plt.plot(xrange, d2[1], label="MBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_2', mvar2[0], v2[4], v2[5], v2[6]), lw=1, ls="-.", color=Q.cOrange[2], alpha=0.7)
    subdivPlot3b, = plt.plot(xrange, d3[1], label="MBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_3', mvar3[0], v3[4], v3[5], v3[6]), lw=1, ls="--", color=Q.cOrange[1], alpha=0.7)
    subdivPlot4b, = plt.plot(xrange, d4[1], label="MBR@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_4', mvar4[0], v4[4], v4[5], v4[6]), lw=1, ls="-", color=Q.cOrange[0], alpha=0.7)
else:
    subdivPlot1, = plt.plot(xrange, d1, label="Sub@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_1', mvar1[0], v1[1], v1[2], v1[3]), lw=1, ls=":", color=Q.cTemporal[3])
    subdivPlot2, = plt.plot(xrange, d2, label="Sub@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_2', mvar2[0], v2[1], v2[2], v2[3]), lw=1, ls="-.", color=Q.cTemporal[2])
    subdivPlot3, = plt.plot(xrange, d3, label="Sub@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_3', mvar3[0], v3[1], v3[2], v3[3]), lw=1, ls="--", color=Q.cTemporal[1])
    subdivPlot4, = plt.plot(xrange, d4, label="Sub@" + Q.genLabel(VAR, MVAR, dmvarStr[MVAR]+'_4', mvar4[0], v4[1], v4[2], v4[3]), lw=1, ls="-", color=Q.cTemporal[0])


#plt.legend(loc="lower left", ncol=2, prop={"size":5.5}, bbox_to_anchor=(-0.2, -0.3))
if measure == "speedup" or measure == "speedup-sbr" or measure == "speedup-mbr":
    plt.legend(prop={"size":5.0}, ncol=2)
else:
    plt.legend(prop={"size":7})


#ax.margins(0.1)
setYscale[dFunc[measure]](ax)
ax.set_xlim([xmin, xmax])
if measure == "speedup" or measure == "wrf":
    setYlim[dFunc[measure]](ax,ymin,ymax, min(Q.minmaxFromLists(min, d1[0], d2[0], d3[0], d4[0], d5[0]), Q.minmaxFromLists(min, d1[1], d2[1], d3[1], d4[1], d5[1])), max(Q.minmaxFromLists(max, d1[0],d2[0],d3[0],d4[0],d5[0]), Q.minmaxFromLists(max, d1[1],d2[1],d3[1],d4[1],d5[1])))
    if MVAR == "A":
        ax.set_yticks([1, mvar1[0], mvar2[0], mvar3[0], mvar4[0]])
        ax.set_yticklabels(['', '${\mathcal{A}_1}$', '${\mathcal{A}_2}$', '${\mathcal{A}_3}$', '${\mathcal{A}_4}$'])
    else:
        ax.set_yticks([1, A[0]])
        ax.set_yticklabels(['', r'$\mathcal{A}$'])
    plt.plot(xrange, A, lw=0.5, ls='--', color=Q.cGrayscale[2])
else:
    setYlim[dFunc[measure]](ax,ymin,ymax, Q.minmaxFromLists(min, d1,d2,d3,d4,d5), Q.minmaxFromLists(max, d1,d2,d3,d4,d5))
plt.tight_layout()
plt.savefig(f'../plots/theo-{measure}-{mode}-multi{MVAR}-{VAR}.pdf', format='pdf')
#plt.show()
print("END\n\n")
