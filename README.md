# GPU Dynamic Parallelism
This program serves as an experimental tool for comparing the efficiency of the GPU under heterogeneous parallel workloads. It uses the 
Mandelbrot Set as case study, offering three GPU-based approaches:
- Exhaustive approach:      The classic flat-kernel parallel 
- CUDA Dynamic Parallelism: Nvidia's recursive kernel approach
- Adaptive Serial Kernels:  An alternative iterative-kernel approach (proposed) 

