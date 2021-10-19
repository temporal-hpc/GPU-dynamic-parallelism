#pragma once

/** CUDA check macro */
#define cucheck(call)                                                               \
    {                                                                               \
        cudaError_t res = (call);                                                   \
        if (res != cudaSuccess) {                                                   \
            const char *err_str = cudaGetErrorString(res);                          \
            fprintf(stderr, "%s (%d): %s in %s\n", __FILE__, __LINE__, err_str,     \
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

/** block size along */
#ifndef BSX
#define BSX 16
#endif

#ifndef BSY
#define BSY 16
#endif

#ifndef REALIZATIONS
#define REALIZATIONS 10
#endif
