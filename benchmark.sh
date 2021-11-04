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
OUTPUT=data/benchmark-REA${REAL}-REP${REPE}-BS${BS}.dat

# COMPILE
make -B REALIZATIONS=${REAL}  REPEATS=${REPE} BSX=${BS} BSY=${BS}
echo "#N,MAXDWELL,MAXDEPTH,   g0,r,B,            perfAP0          perfA1       perfA2       perfA3    (${DATE})">> ${OUTPUT}

# RUN
for g0 in 2 4 8 16 32 64 128 256 512 1024
do
    for r in 2 4 8 16 32 64 128 256 512 1024
    do
        for B in 1 2 4 8 16 32 64 128 256 512 1024
        do
            for size in {5..16..1}
            do
                N=$((2**${size}))
                a="${N},    ${CA_MAXDWELL},${MAX_DEPTH},   ${g0},${r},${B}"
                echo "BENCHMARK g0=${g0},r=${r},B=${B},CA_MAXDWELL=${CA_MAXDWELL},MAX_DEPTH=${MAX_DEPTH},    N=${N}"
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
