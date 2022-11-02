#!/bin/sh
if [ "$#" -ne 2 ]; then
    echo "run as ./genplots.sh <measure> <mode>"
    echo "<measure> = {work, wrf, time, speedup}"
    echo "<mode> = {optimal, fixed}"
    exit
fi

# chosen measure (work, wrf, time,speedup)
mes=${1}
mode=${2}

# fixed variables
# original
#cn=$((2**30)); cg=8; cB=8; cr=2; cP=0.7; clam=100; cA=16;

# testing for SBR better than MBR
cn=$((2**24)); cg=8; cB=8; cr=2; cP=0.7; clam=100; cA=8;

# parallelism parameters
cq=128; cc=64;
# number of samples
res=$((2**10))
# legend parameters
n1=$((2**8)); n2=$((2**16)); n3=$((2**24)); n4=$((2**32))
#n1=$((2**10)); n2=$((2**12)); n3=$((2**14)); n4=$((2**16))
g1=128; g2=64; g3=32; g4=2
B1=4; B2=16; B3=64; B4=256
r1=64; r2=32; r3=8; r4=2
P1=0.9; P2=0.8; P3=0.7; P4=0.6
lam1=$((10**6)); lam2=$((10**4)); lam3=$((10**2)); lam4=$((10**0))
#lam1=$((10**3)); lam2=$((10**2)); lam3=$((10**1)); lam4=$((10**0))
A1=2; A2=4; A3=8; A4=16
q1=512; q2=256; q3=128; q4=1
c1=128; c2=64; c3=16; c4=1
# VAR default ranges
nx1=1;      nx2=$((2**33))
gx1=2;      gx2=$((1024))
Bx1=1;      Bx2=$((2**25))
rx1=2;      rx2=$((1024))
Px1=0;     Px2=1
lamx1=1;    lamx2=$((10**6))
Ax1=1;      Ax2=$((2**20))
qx1=1;      qx2=$((2**20))
cx1=1;      cx2=$((2**20))
ymin=0
#ymax=$((${cA}+5))
aux=$(echo "scale=2;${cA}*1.6" |bc)
#echo "-----------------------------------"
ymax=${aux%.*}
#echo $ymax
#echo "-----------------------------------"
multiVars=("q" "c" "lam" "n" "g" "B" "r" "P" "A")
variables=("q" "c" "lam" "g" "B" "r" "P" "A" "n")
na=("n" ${n1} ${n2} ${n3} ${n4} ${nx1} ${nx2})
ga=("g" ${g1} ${g2} ${g3} ${g4} ${gx1} ${gx2})
Ba=("B" ${B1} ${B2} ${B3} ${B4} ${Bx1} ${Bx2})
ra=("r" ${r1} ${r2} ${r3} ${r4} ${rx1} ${rx2})
Pa=("P" ${P1} ${P2} ${P3} ${P4} ${Px1} ${Px2})
lama=("lam" ${lam1} ${lam2} ${lam3} ${lam4} ${lamx1} ${lamx2})
Aa=("A" ${A1} ${A2} ${A3} ${A4} ${Ax1} ${Ax2})
qa=("q" ${q1} ${q2} ${q3} ${q4} ${qx1} ${qx2})
ca=("c" ${c1} ${c2} ${c3} ${c4} ${cx1} ${cx2})

CONFIG="${mes}     ${cn} ${cg} ${cB} ${cr} ${cP}  ${clam}  ${cA}  ${cq} ${cc}"
SUFFIX="${ymin}   $((${ymax}+1))    ${res}    ${mode}"
for i in "${multiVars[@]}"; do
    declare -n mul="${i}a"
    PREFIX="${CONFIG}       ${mul[1]} ${mul[2]} ${mul[3]} ${mul[4]}     ${i}"
    for j in "${variables[@]}"; do
        declare -n xvars="${j}a"
        if [ "${i}" != "${j}" ]
        then
            echo "python cmodel.py ${PREFIX} ${j} ${xvars[5]}  ${xvars[6]}  ${SUFFIX}"
            python cmodel.py ${PREFIX} ${j} ${xvars[5]}  ${xvars[6]}  ${SUFFIX}
        fi
    done
done
