/*
 Copyright (c) 2022 Temporal Guild Group, Austral University of Chile, Valdivia Chile.
 This file and all powermon software is licensed under the MIT License.
 Please refer to LICENSE for more details.
 */
#include "nvmlPower.hpp"

double gpuCurrentPower;
double gpuAveragePower;
double gpuTotalEnergy;
double gpuTotalTime;

bool GPUpollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];
std::string GPUfilename;

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode;
nvmlComputeMode_t computeMode;

int GPU_SAMPLE_US = 5000;

pthread_t GPUpowerPollThread;

/*
Poll the GPU using nvml APIs.
*/
void* GPUpowerPollingFunc(void* ptr) {

    unsigned int powerLevel = 0;
    FILE* fp = fopen(GPUfilename.c_str(), "w+");
    int timestep = 0;
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    double dt = 0.0;
    double acctime = 0.0;
    double accenergy = 0.0;
    double power = 0.0;
    // column names
    fprintf(fp, "%-15s, %-15s, %-15s, %-15s, %-15s, %-15s,\n", "#timestep", "power", "acc-energy", "avg-power", "dt", "acc-time");

    while (GPUpollThreadStatus) {
        timestep++;
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
        usleep(GPU_SAMPLE_US);
        gettimeofday(&t2, NULL);
        dt = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1000000.0);
        acctime += dt;
        // Get the power management mode of the GPU.
        // nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

        // The following function may be utilized to handle errors as needed.
        // getNVMLError(nvmlResult);

        // Check if power management mode is enabled.
        // if (pmmode == NVML_FEATURE_ENABLED){
        // Get the power usage in milliWatts.
        nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
        //}
        power = (double)powerLevel / 1000.0;
        gpuCurrentPower = power;
        accenergy += power * dt;
        // The output file stores power in Watts.
        fprintf(fp, "%-15i %-15f %-15f %-15f %-15f %-15f\n", timestep, power, accenergy, accenergy / acctime, dt, acctime);
        t1 = t2;
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    }
    fclose(fp);
    gpuAveragePower = accenergy / acctime;
    gpuTotalEnergy = accenergy;
    gpuTotalTime = acctime;
    pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void GPUPowerBegin(size_t N, int us, int comptype, std::string strr) {
    GPU_SAMPLE_US = us;
    int i;
    // Initialize nvml.
    nvmlResult = nvmlInit();
    if (NVML_SUCCESS != nvmlResult) {
        printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }

    // Count the number of GPUs available.
    nvmlResult = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }

    for (i = 0; i < deviceCount; i++) {
        // Get the device ID.
        nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
        if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }
        // Get the name of the device.
        nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr) / sizeof(deviceNameStr[0]));
        if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }
        // Get PCI information of the device.
        nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
        if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }
        // Get the compute mode of the device which indicates CUDA capabilities.
        nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
        if (NVML_ERROR_NOT_SUPPORTED == nvmlResult) {
            printf("This is not a CUDA-capable device.\n");
        } else if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }
    }

    // This statement assumes that the first indexed GPU will be used.
    // If there are multiple GPUs that can be used by the system, this needs to be done with care.
    // Test thoroughly and ensure the correct device ID is being used.
    nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);
    GPUpollThreadStatus = true;
    GPUfilename = std::string("data/GPUPOWER-") + std::to_string(N) + "-" + std::to_string(comptype) + "-" + strr + std::string(".dat");
    const char* message = "GPUpower";
    int iret = pthread_create(&GPUpowerPollThread, NULL, GPUpowerPollingFunc, (void*)message);
    if (iret) {
        fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
        exit(0);
    }
    usleep(1000 * WARMUP_MS);
}

/*
End power measurement. This ends the polliing thread.
*/
void GPUPowerEnd() {
    usleep(1000 * COOLDOWN_MS);
    GPUpollThreadStatus = false;
    pthread_join(GPUpowerPollThread, NULL);

    nvmlResult = nvmlShutdown();
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int getNVMLError(nvmlReturn_t resultToCheck) {
    if (resultToCheck == NVML_ERROR_UNINITIALIZED)
        return 1;
    if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
        return 2;
    if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
        return 3;
    if (resultToCheck == NVML_ERROR_NO_PERMISSION)
        return 4;
    if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
        return 5;
    if (resultToCheck == NVML_ERROR_NOT_FOUND)
        return 6;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
        return 7;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
        return 8;
    if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
        return 9;
    if (resultToCheck == NVML_ERROR_TIMEOUT)
        return 10;
    if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
        return 11;
    if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
        return 12;
    if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
        return 13;
    if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
        return 14;
    if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
        return 15;
    if (resultToCheck == NVML_ERROR_UNKNOWN)
        return 16;

    return 0;
}
