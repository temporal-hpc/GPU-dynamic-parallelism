#!/bin/bash
./exp-GPU-results.sh ../data/A100-ARCHsm_80 "A100"                    16 16    64  8  64  8  16 16  16 16    108 64    $((2**16))    2 2 4    0 15
./exp-GPU-results.sh ../data/TITAN-RTXsm_75 "TITAN-RTX"               16 16    64  8  64  8  16 16  64  4     72 64    $((2**16))    2 2 4    0 15
./exp-GPU-results.sh ../data/TITAN-Vsm_70 "TITAN-V"                    8  8    64  8  64  8  16 16  64  4     80 64    $((2**15))    2 2 4    0 15
./exp-GPU-results.sh ../data/JETSON-XAVIER-NXsm_72 "JETSON-XAVIER-NX" 64  4    16 16  16 16   8  8  16 16      6 64    $((2**14))    2 2 4    0 15
