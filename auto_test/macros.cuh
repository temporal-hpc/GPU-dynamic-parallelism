#pragma once

/** DEBUG macro **/
#ifdef DEBUG
#define D(x) (x)
#else
#define D(x)                                                                        \
    do {                                                                            \
    } while (0)
#endif

/** CUDA check macro */
#define cucheck(call)                                                               \
    {                                                                               \
        cudaError_t res = (call);                                                   \
        if (res != cudaSuccess) {                                                   \
            const char *err_str = cudaGetErrorString(res);                          \
            fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str,       \
                    #call);                                                         \
            exit(22);                                                               \
        }                                                                           \
    }

#define cucheck_dev(call)                                                           \
    {                                                                               \
        cudaError_t res = (call);                                                   \
        if (res != cudaSuccess) {                                                   \
            const char *err_str = cudaGetErrorString(res);                          \
            printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);        \
            assert(0);                                                              \
        }                                                                           \
    }

// Some cuda helper functions.
void DisplayHeader() {
    const int kb = 1024;
    const int mb = kb * kb;
    D(wcout << "CUDA version:   v" << CUDART_VERSION << endl;)
    int devCount;
    cudaGetDeviceCount(&devCount);
    D(wcout << "CUDA Devices: " << endl;)
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        D(wcout << i << ": " << props.name << endl;)
    }
}

void size_t getFreeMemory() {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        D(cout << "GPU " << id << " memory: free=" << free << ", total=" << total
               << endl);
    }
    return free;
}

/** block size along */
#ifndef BSX
#define BSX 16
#endif

#ifndef BSY
#define BSY 16
#endif