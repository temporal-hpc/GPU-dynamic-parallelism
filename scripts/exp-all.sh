#!/bin/bash
echo "./exp-genplots-GPU.sh ../data/A100-ARCHsm_80 "NVIDIA A100"              16 16    64  8  64  8  16 16  16 16    108 64    $((2**16))    2 2 4    0 16"
./exp-genplots-GPU.sh ../data/A100-ARCHsm_80 "NVIDIA A100"                    16 16    64  8  64  8  16 16  16 16    108 64    $((2**16))    2 2 4    0 16

echo "./exp-genplots-GPU.sh ../data/TITAN-RTX-ARCHsm_75 "NVIDIA TITAN RTX"         8 8    64  8  16  16  16 16  64  4     72 64    $((2**16))    2 2 4    0 16"
./exp-genplots-GPU.sh ../data/TITAN-RTX-ARCHsm_75 "NVIDIA TITAN RTX"               8 8    64  8  16  16  16 16  64  4     72 64    $((2**16))    2 2 4    0 16

echo "./exp-genplots-GPU.sh ../data/TITAN-V-ARCHsm_70 "NVIDIA TITAN V"              8  8    64  8  64  8  16 16  64  4     80 64    $((2**15))    2 2 4    0 16"
./exp-genplots-GPU.sh ../data/TITAN-V-ARCHsm_70 "NVIDIA TITAN V"                    8  8    64  8  64  8  16 16  64  4     80 64    $((2**15))    2 2 4    0 16

echo "./exp-genplots-GPU.sh ../data/JETSON-XAVIER-NX-ARCHsm_72 "NVIDIA JETSON XAVIER NX" 64 4  16 16  16 16 8  8 16 16      6 64    $((2**14))    2 2 4    0 16"
./exp-genplots-GPU.sh ../data/JETSON-XAVIER-NX-ARCHsm_72 "NVIDIA JETSON XAVIER NX"       64 4  16 16  16 16 8  8 16 16      6 64    $((2**14))    2 2 4    0 16
