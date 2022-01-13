import numpy as np
import sys

if len(sys.argv) != 2:
    print(f"run as ./plot-results.py <data-file>\n")
    exit()

dframe = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=2)
print(dframe.shape)
print(dframe[dframe[:,0]==4])
m1 = dframe[:,1]==8
m2 = dframe[:,2]==8
m3 = dframe[:,3]==2
mask = m1 & m2 & m3
print(dframe[mask])
