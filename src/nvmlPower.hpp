/*
 Copyright (c) 2021 Temporal Guild Group, Austral University of Chile, Valdivia Chile.
 This file and all powermon software is licensed under the MIT License.
 Please refer to LICENSE for more details.
 */
 /*
 Header file including necessary nvml headers.
 */
#ifndef INCLNVML
#define INCLNVML
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <string>
#define COOLDOWN_MS 100
#define WARMUP_MS 	200
extern double gpuCurrentPower;
extern double gpuAveragePower;
extern double gpuTotalEnergy;
extern double gpuTotalTime;
// GPU power measure functions
void GPUPowerBegin(size_t N, int ms, int comptype_gpu, std::string strr);
void GPUPowerEnd();
// pthread functions
void* GPUpowerPollingFunc(void* ptr);
int getNVMLError(nvmlReturn_t resultToCheck);
#endif
