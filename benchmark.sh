#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "run as ./benchmark.sh <REALIZATIONS> <REPEATS> <BS>"
    exit
fi
REAL=$1
REPE=$2
BS=${3}
echo "REALIZATIONS=${REAL}  REPEATS=${REPE}"
GPUPROG=./bin/gpuDP
CA_MAXDWELL=512
MAX_DEPTH=1000
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "DATE = ${DATE}"
OUTPUT=data/benchmark-REA${REAL}-REP${REPE}-BS${BS}-MAXDWELL${CA_MAXDWELL}-MAXDEPTH${MAX_DEPTH}.dat

# COMPILE
make -B REALIZATIONS=${REAL}  REPEATS=${REPE} BSX=${BS} BSY=${BS}
echo "NEW BENCHMARK ${DATE}">> ${OUTPUT}
echo "#N, g0,r,B,                   perfAP0                             perfA1                            perfA2                               perfA3" >> ${OUTPUT}

maxEXP=10
NexpMAX=16

# RUN
for ((size=0; size <= ${NexpMAX}; size++));
do
    N=$((2**${size}))
    #for g0exp in 2 4 8 16 32 64 128 256 512 1024
    lim=$((${size}<${maxEXP} ? ${size} : ${maxEXP}))
    for ((g0exp=1; g0exp <= ${lim}; g0exp++));
    do
        g0=$((2**${g0exp}))
        aux=$((${size}-${g0exp}))
        divlimit=$((${aux}<${maxEXP} ? ${aux} : ${maxEXP}))
        #for r in 2 4 8 16 32 64 128 256 512 1024
        #for rexp in {1..${divlimit}..1}
        for ((rexp=1; rexp <= ${divlimit}; rexp++));
        do
            r=$((2**${rexp}))
            #for B in 1 2 4 8 16 32 64 128 256 512 1024
            #for Bexp in {0..${divlimit}..1}
            for ((Bexp=0; Bexp <= ${divlimit}; Bexp++));
            do
                B=$((2**${Bexp}))
                a="${N},    ${g0},${r},${B}"
                echo "BENCHMARK g0=${g0},r=${r},B=${B},    N=${N}"
                for approach in 0 1 2 3
                do
                    echo -n -e "\tA${approach}......"
                    res=$(exec ${GPUPROG} $approach ${N} ${N} -1.5 0.5 -1 1 ${CA_MAXDWELL} ${B} ${g0} ${r} $MAX_DEPTH none)
                    echo "done: ${res}"
                    a="${a},        ${res}"
                done
                echo $a >> ${OUTPUT}
                echo $a
                echo -e "\n"
            done
        done
    done
done
