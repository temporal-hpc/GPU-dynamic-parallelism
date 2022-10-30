#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "run as ./measurePower.sh <DEV> <ARCH> <SIZE>"
    echo "Example: ./measurePower.sh 0 sm_80 \$((2**32))"
    exit
fi
dev=$1
arch=$2
size=$3
#         ⮟EX    ⮟DP-SBR   ⮟DP-MBR    ⮟ASK-SBR  ⮟ADK-MBR
configs=("0 0 0" "64 4 32"  "64 4 64"  "32 4 16"  "32 2 32")    ## TITAN V
#configs=("0 0 0" "64 4 128"  "32 4 64"  "64 4 16"  "32 2 32")   ## TITAN RTX
#configs=("0 0 0" "16 2 128"  "16 4 128"  "64 4 16"  "64 2 32")  ## A100
#configs=("0 0 0" "32 4 64"  "32 4 64"  "32 4 16"  "64 2 32")    ## JETSEN

for app in {0..4}
do
    conf=${configs[$app]}
    echo $app
    for i in "8 8" "16 16" "64 4" "64 8" "32 32"
    do 
        set -- $i
        make ARCH=${arch} MEASURE_POWER=MEASURE_POWER BSX=${1} BSY=${2} REPEATS=100 REALIZATIONS=1
        echo "./bin/gpuDP ${dev} ${app} ${size} ${size} -1.5 0.5 -1.0 1.0 512 ${conf} 1000 none"
        ./bin/gpuDP ${dev} ${app} ${size} ${size} -1.5 0.5 -1.0 1.0 512 ${conf} 1000 none
        echo $x
    done
done
echo $arch
