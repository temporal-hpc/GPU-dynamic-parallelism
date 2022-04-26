#!/bin/bash
DATAPATH=${1}; GPUNAME=${2}
BSX0=${3}; BSY0=${4}
BSX1=${5}; BSY1=${6}
BSX2=${7}; BSY2=${8}
BSX3=${9}; BSY3=${10}
BSX4=${11}; BSY4=${12}
HWQ=${13}; HWC=${14}
MAXN=${15}
g=${16}; r=${17}; B=${18}
miny=${19}; maxy=${20}

echo "GPUNAME=${GPUNAME}"

# python explot command
#python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} time ${MAXN} ${g} ${r} ${B} n optimal ${miny} ${maxy}
python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} speedup ${MAXN} ${g} ${r} ${B} grB optimal ${miny} ${maxy}
python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} speedup ${MAXN} ${g} ${r} ${B} n optimal ${miny} ${maxy}
python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} speedup ${MAXN} ${g} ${r} ${B} g optimal ${miny} ${maxy}
python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} speedup ${MAXN} ${g} ${r} ${B} r optimal ${miny} ${maxy}
python explot.py ${DATAPATH} "${GPUNAME}" ${BSX0} ${BSY0} ${BSX1} ${BSY1} ${BSX2} ${BSY2} ${BSX3} ${BSY3} ${BSX4} ${BSY4}   ${HWQ} ${HWC} speedup ${MAXN} ${g} ${r} ${B} B optimal ${miny} ${maxy}
