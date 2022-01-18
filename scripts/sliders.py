import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import tools as Q

def updatePlot():
    dataSubdiv = funcs[dFunc[measure]](n,B,s,P,lam,A)
    subdivPlot.set_ydata(dataSubdiv)

    dataExhaustive = refFuncs[dFunc[measure]](n,B,s,P,lam,A)
    refPlot.set_ydata(dataExhaustive)

    ax.set_ylim([0, dataSubdiv.max()])
    fig.canvas.draw_idle()


# The function to be called anytime a slider's value changes
def updaten(val):
    global n
    n = np.full(res, val)
    updatePlot()
def updateB(val):
    global B
    B = np.full(res, val)
    updatePlot()
def updateP(val):
    global P
    P = np.full(res, val)
    updatePlot()
def updateS(val):
    global s
    s = np.full(res, val)
    updatePlot()
def updatelam(val):
    global lam
    lam = np.full(res, val)
    updatePlot()
def updateA(val):
    global A
    A = np.full(res, val)
    updatePlot()

def addSlider(posy, name, min, max, default, func, sliderFormat):
    ax = plt.axes([0.12, posy, 0.7, 0.05])
    slider = createSlider(ax, name, min, max, default, sliderFormat)
    slider.on_changed(func)
    return slider

def createSlider(ax, label, valmin, valmax, valinit, sliderFormat):
    return Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit, valfmt=sliderFormat)


# ------------
# main code
# ------------
if len(sys.argv) !=8:
    print("\nEjecutar como python <prog> <measure> <VAR> <xmin> <xmax> <ymin> <ymax> <res>")
    print("measure = {work, wrf}")
    print("VAR = {n, P, lam, B, s}")
    print("res = number of points in [xmin, xmax]")
    exit(2)

dFunc = {"work":0, "wrf":1}
dLabel = {"work":r"${W}$", "wrf":r"${\dfrac{W_{BF}}{W}}$"}
dTitle = {"work":"Work", "wrf":"Work Reduction Factor"}
# parameters
res = int(sys.argv[7])
n = np.full(res, 4096)
B = np.full(res, 16)
P = np.full(res, 0.95)
lam = np.full(res, 0.1)
A = np.full(res, 4)
s = np.full(res, 2)
measure = sys.argv[1]
mi = dFunc[measure]
VAR = sys.argv[2]

# x range
xmin = int(sys.argv[3])
xmax = int(sys.argv[4])

# y range
ymin = int(sys.argv[5])
ymax = int(sys.argv[6])

# creating xrange
xrange = np.linspace(xmin, xmax, res)

print(f"dFunc[{measure}] = {dFunc[measure]}")
#print(f"dLabel[{measure}] = {dLabel[measure]}")
#input("press enter")
funcs = [lambda n,B,s,P,lam,A: Q.subdiv(n, B, s, P, lam, A),lambda n,B,s,P,lam,A: Q.exhaustive(n,A)/Q.subdiv(n, B, s, P, lam, A)]
refFuncs = [lambda n,B,s,P,lam,A: Q.exhaustive(n,A), lambda n,B,s,P,lam,A: Q.exhaustive(n,A)/Q.exhaustive(n,A)]
# creating plot
print(f"VAR = {VAR}, range {xmin} -> {xmax}")
fig, ax = plt.subplots()
plt.title(dTitle[measure] + f' vs {VAR}')
ax.set_xlabel(f'{VAR}')
ax.set_ylabel(dLabel[measure], rotation=0, labelpad=15)

if VAR == "n":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    n = xrange
    ax.set_xscale('log', base=2)
if VAR == "B":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    B = xrange
    ax.set_xscale('log', base=2)
if VAR == "P":
    P = xrange
if VAR == "s":
    xrange = np.logspace(np.log2(xmin), np.log2(xmax), res, base=2)
    s = xrange
    ax.set_xscale('log', base=2)
if VAR == "lam":
    lam = xrange

d1 = funcs[dFunc[measure]](n,B,s,P,lam,A)
d2 = refFuncs[dFunc[measure]](n,B,s,P,lam,A)
subdivPlot, = plt.plot(xrange, d1,      label="Subdivision", lw=2)
refPlot,    = plt.plot(xrange, d2,    label="Exhaustive", lw=2, ls=':')
plt.legend()
ax.margins(0.1)
ax.set_xlim([xmin, xmax])
#ax.set_ylim([ymin, ymax])
ax.set_ylim([0, d1.max()])

# adjust the main plot to make room for the sliders
plt.subplots_adjust(bottom=0.5)

offset = 0.30
if VAR != "n":
    s0 = addSlider(offset, 'n', 1, 32768, n[0], updaten, "%i")
    offset -= 0.06
if VAR != "B":
    s1 = addSlider(offset, 'B', 1, 256, B[0], updateB, "%i")
    offset -= 0.06
if VAR != "P":
    s2 = addSlider(offset, 'P', 0,     1, P[0], updateP, "%.3f")
    offset -= 0.06
if VAR != "s":
    s3 = addSlider(offset, 's', 2,   256, s[0], updateS, "%i")
    offset -= 0.06
if VAR != "lam":
    s4 = addSlider(offset, 'lam', 0,  1000, lam[0], updatelam, "%.3f")
    offset -= 0.06
s5 = addSlider(offset, 'A', 1, 100, A[0], updateA, "%.3f")
plt.show()
print("END")
