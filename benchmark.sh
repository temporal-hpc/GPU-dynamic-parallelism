#!/bin/bash
if [ "$#" -ne 5 ]; then
    echo "run as ./benchmark.sh <STRING> <ARCH> <REALIZATIONS> <REPEATS> <BS>"
    echo "Example: ./benchmark.sh A100 sm_80 4 8 32"
    exit
fi
STRING=$1
ARCH=$2
REAL=$3
REPE=$4
BS=$5
echo "REALIZATIONS=${REAL}  REPEATS=${REPE}"
GPUPROG=./bin/gpuDP
CA_MAXDWELL=512
MAX_DEPTH=1000
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "DATE = ${DATE}"
OUTPUT=data/${STRING}-ARCH${ARCH}-REA${REAL}-REP${REPE}-BS${BS}.dat

# COMPILE
make -B ARCH=${ARCH} REALIZATIONS=${REAL}  REPEATS=${REPE} BSX=${BS} BSY=${BS} BENCHMARK=BENCHMARK
echo "#NEW BENCHMARK ${STRING} ${ARCH}  ${DATE}        MAXDWELL=${CA_MAXDWELL}  MAX_DEPTH=${MAX_DEPTH}">> ${OUTPUT}
echo "#N, g,r,B,                   perfAP0                             perfA1                            perfA2                               perfA3" >> ${OUTPUT}

maxEXP=10
NexpMAX=16

# RUN
for ((size=0; size <= ${NexpMAX}; size++));
do
    N=$((2**${size}))
    lim=$((${size}<${maxEXP} ? ${size} : ${maxEXP}))
    for ((gexp=1; gexp <= ${lim}; gexp++));
    do
        g=$((2**${gexp}))
        aux=$((${size}-${gexp}))
        divlimit=$((${aux}<${maxEXP} ? ${aux} : ${maxEXP}))
        for ((rexp=1; rexp <= ${divlimit}; rexp++));
        do
            r=$((2**${rexp}))
            for ((Bexp=0; Bexp <= ${divlimit}; Bexp++));
            do
                B=$((2**${Bexp}))
                a="${N},    ${g},${r},${B}"
                echo "BENCHMARK g=${g},r=${r},B=${B},    N=${N}"
                for approach in 0 1 2 3
                do
                    echo -n -e "\tA${approach}......"
                    res=$(exec ${GPUPROG} $approach ${N} ${N} -1.5 0.5 -1 1 ${CA_MAXDWELL} ${g} ${r} ${B} $MAX_DEPTH none)
                    echo "done: ${res}"
                    a="${a},        ${res}"
                done
                echo $a >> ${OUTPUT}
                echo -e "\n"
            done
        done
    done
done
