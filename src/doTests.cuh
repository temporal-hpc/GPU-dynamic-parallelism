#pragma once

float doExhaustive(int *d_dwells, unsigned int w, unsigned int h,
                   complex bottomLeftCorner, complex upperRightCorner,
                   unsigned int CA_MAXDWELL) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(divup(w, blockSize.x), divup(h, blockSize.y));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner,
                                              upperRightCorner, CA_MAXDWELL);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}


float doAdaptiveSerialKernels(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernels(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doAdaptiveSerialKernelsNEW(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernelsNEW(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doDynamicParallelism(int *d_dwells, unsigned int w, unsigned int h,
                           complex bottomLeftCorner, complex upperRightCorner,
                           unsigned int G0, unsigned int r,
                           unsigned int CA_MAXDWELL, unsigned int B,
                           unsigned int MAX_DEPTH) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(G0, G0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_block_k<<<gridSize, blockSize>>>(
            d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0,
            w / G0, 1, r, CA_MAXDWELL, B, MAX_DEPTH);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doGridLines(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        GridLines(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

