/** @file histo-global.cu histogram with global memory atomics */

#include <assert.h>
#include <png.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>    // std::swap


/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}

#define cucheck_dev(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	assert(0);																												\
	}\
	}

/** time spent in device */
double gpu_time = 0;

/** a useful function to compute the number of threads */
__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell);

/** save the dwell into a PNG file 
		@remarks: code to save PNG file taken from here 
		  (error handling is removed):
		http://www.labbookpages.co.uk/software/imgProc/libPNG.html
 */
void save_image(const char *filename, int *dwells, int w, int h) {
	png_bytep row;
	
	FILE *fp = fopen(filename, "wb");
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	// exception handling
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	// write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, w, h,
							 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
							 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	// set title
	png_text title_text;
	title_text.compression = PNG_TEXT_COMPRESSION_NONE;
	title_text.key = "Title";
	title_text.text = "Mandelbrot set, per-pixel";
	png_set_text(png_ptr, info_ptr, &title_text, 1);
	png_write_info(png_ptr, info_ptr);

	// write image data
	row = (png_bytep) malloc(3 * w * sizeof(png_byte));
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int r, g, b;
			dwell_color(&r, &g, &b, dwells[y * w + x]);
			row[3 * x + 0] = (png_byte)r;
			row[3 * x + 1] = (png_byte)g;
			row[3 * x + 2] = (png_byte)b;
		}
		png_write_row(png_ptr, row);
	}
	png_write_end(png_ptr, NULL);

  fclose(fp);
  png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  free(row);
}  // save_image


/** a simple complex type */
struct complex {
	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}
	/** real and imaginary part */
	float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex &a) { return complex(-a.re, -a.im); }
inline __host__ __device__ complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
}  // operator/

#ifndef MAX_DWELL
#define MAX_DWELL 512
#endif
/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#ifndef MAX_DEPTH
#define MAX_DEPTH 5
#endif
/** region below which do per-pixel */
#ifndef MIN_SIZE
#define MIN_SIZE 32
#endif
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32

/** find the dwell for the pixel */
__device__ int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y) {
	complex dc = cmax - cmin;
	float fx = (float)x / w, fy = (float)y / h;
	complex c = cmin + complex(fx * dc.re, fy * dc.im);
	int dwell = 0;
	complex z = c;
	while(dwell < MAX_DWELL && abs2(z) < 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}  // pixel_dwell

/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
		element, -1 = dwells are different */
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__ int same_dwell(int d1, int d2) {
	if(d1 == d2)
		return d1;
	else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
		return min(d1, d2);
	else
		return DIFF_DWELL;
}  // same_dwell


/** checking for an error */
__device__ void check_error(int x0, int y0, int d) {
	int err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("error launching kernel for region (%d..%d, %d..%d)\n", 
					 x0, x0 + d, y0, y0 + d);
		assert(0);
	}
}

__global__ void border_dwell2
(int* d_ns, int* d_offs1, int* d_offs2, int* dwells, int w, int h, complex cmin, complex cmax, int d, int depth, int subdiv) {
	// check whether all boundary pixels have the same dwell
    int use = ((blockIdx.x << 4) + blockIdx.z*gridDim.y+blockIdx.y)<<1;
    int x0 = d_offs1[use];
    int y0 = d_offs1[use + 1];
    
    //int x0 = d_offs[(blockIdx.x)*2];
    //int y0 = d_offs[(blockIdx.x)*2 + 1];
    __shared__ int off_index;

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int bs = blockDim.x * blockDim.y;
	int comm_dwell = NEUT_DWELL;
	// for all boundary pixels, distributed across threads
	for(int r = tid; r < d; r += bs) {
		// for each boundary: b = 0 is east, then counter-clockwise
		for(int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
			comm_dwell = same_dwell(comm_dwell, dwell);
            //dwells[y * w + x] = 512;//pixel_dwell(w, h, cmin, cmax, x, y);
		}
	}  // for all boundary pixels
	// reduce across threads in the block
	__shared__ int ldwells[BSX * BSY];
	int nt = min(d, BSX * BSY);
	if(tid < nt)
		ldwells[tid] = comm_dwell;
	__syncthreads();
	for(; nt > 1; nt /= 2) {
		if(tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
		__syncthreads();
	}
    comm_dwell = ldwells[0];

    if(comm_dwell != DIFF_DWELL) {
        //return;
        
        int x = threadIdx.x;
        int y = threadIdx.y;
        for (int ry=y; ry < d; ry+=blockDim.y){
            for (int rx=x; rx < d; rx+=blockDim.x){
                if(rx < d && ry < d) {
                    int rxx = rx+x0, ryy = ry+y0;
                    dwells[ryy * w + rxx] = comm_dwell;
                }

            }
        }
    } else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
        if (tid == 0){
            off_index = atomicAdd(d_ns, 1);
            //printf("%i\n", off_index);
        }
        __syncthreads();
        if (tid < 32){
            d_offs2[/* 32 de 2*subdiv*subdiv */ (off_index<<5)+tid] = (x0 + ((tid >> 1) & 3)*d/SUBDIV)*((tid+1) & 1) + (y0 + (tid >> 3)*d/SUBDIV)*((tid)&1);
        }
    } else {
        //return;
        int x = threadIdx.x ;
        int y = threadIdx.y ;
        for (int ry=y; ry < d; ry+=blockDim.y){
            for (int rx=x; rx < d; rx+=blockDim.x){
                if(rx < d && ry < d) {
                    int rxx = rx+x0, ryy = ry+y0;
                    dwells[ryy * w + rxx] = pixel_dwell(w, h, cmin, cmax, rxx, ryy);
                }

            }
        }
    }
    //cucheck_dev(cudaGetLastError());

}  // border_dwell

void mandelbrot_pseudo_dynamic_parallelism(int *dwell, int* h_nextSize, int* d_nextSize, int* d_offsets1, int* d_offsets2, int w, int h, complex cmin, complex cmax, int d, int depth){
    
	dim3 b(BSX, BSY, 1), g(1, INIT_SUBDIV, INIT_SUBDIV);
    //printf("Running kernel with b(%i,%i) and g(%i, %i, %i) and d=%i\n", b.x, b.y, g.x, g.y, g.z, d);
    border_dwell2<<<g, b>>>(d_nextSize, d_offsets1, d_offsets2, dwell, h, w, cmin, cmax, d, depth, INIT_SUBDIV);
    for (int i=depth+1; i<MAX_DEPTH; i++){
        cudaMemcpy(h_nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost);
        (cudaMemset(d_nextSize, 0, sizeof(int)));
        std::swap(d_offsets1, d_offsets2);
        d = d >> 2;
        dim3 g(*h_nextSize, SUBDIV, SUBDIV);
        //printf("Running kernel with b(%i,%i) and g(%i, %i, %i) and d=%i\n", b.x, b.y, g.x, g.y, g.z, d);
        border_dwell2<<<g, b>>>(d_nextSize, d_offsets1, d_offsets2, dwell, h, w, cmin, cmax, d, i, SUBDIV);
    }


}

/** gets the color, given the dwell (on host) */
#define CUT_DWELL (MAX_DWELL / 4)
void dwell_color(int *r, int *g, int *b, int dwell) {
	// black for the Mandelbrot set
	if(dwell >= MAX_DWELL) {
		*r = *g = *b = 0;
	} else {
		// cut at zero
		if(dwell < 0)
			dwell = 0;
		if(dwell <= CUT_DWELL) {
			// from black to blue the first half
			*r = *g = 0;
			*b = 128 + dwell * 127 / (CUT_DWELL);
		} else {
			// from blue to white for the second half
			*b = 255;
			*r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
		}
	}
}  // dwell_color

/** data size */
#ifndef H
#define H (32 * 1024)
#endif 

#ifndef W
#define W (32 * 1024)
#endif

#define IMAGE_PATH "./mandelbrot.png"

int main(int argc, char **argv) {
	// allocate memory
	int w = W, h = H;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	cucheck(cudaMalloc((void**)&d_dwells, dwell_sz));
	h_dwells = (int*)malloc(dwell_sz);

    int *h_nextSize, *h_offsets;
    int *d_nextSize;
    int *d_offsets1;
    int *d_offsets2;
	h_nextSize = (int*)malloc(sizeof(int));
	h_offsets = (int*)malloc(sizeof(int)*2*(INIT_SUBDIV*INIT_SUBDIV)*pow(SUBDIV*SUBDIV, MAX_DEPTH-1));
    for (int i=0; i<INIT_SUBDIV*INIT_SUBDIV*2; i+=2){
        h_offsets[i] = ((i/2)%INIT_SUBDIV)*(W/INIT_SUBDIV);
        h_offsets[i+1] = ((i/2)/INIT_SUBDIV)*(W/INIT_SUBDIV);

        //printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i], h_offsets[i+1]);
        
    }
    *h_nextSize = INIT_SUBDIV*INIT_SUBDIV;
	cucheck(cudaMalloc((void**)&d_nextSize, sizeof(int)));
	cucheck(cudaMalloc((void**)&d_offsets1, sizeof(int)*2*(INIT_SUBDIV*INIT_SUBDIV)*pow(SUBDIV*SUBDIV, MAX_DEPTH-1)));
	cucheck(cudaMalloc((void**)&d_offsets2, sizeof(int)*2*(INIT_SUBDIV*INIT_SUBDIV)*pow(SUBDIV*SUBDIV, MAX_DEPTH-1)));
    cucheck(cudaMemcpy(d_offsets1, h_offsets, sizeof(int)*2*(INIT_SUBDIV*INIT_SUBDIV)*pow(SUBDIV*SUBDIV, MAX_DEPTH-1), cudaMemcpyHostToDevice))
    cucheck(cudaMemset(d_nextSize, 0, sizeof(int)));
	// compute the dwells, copy them back
	double t1 = omp_get_wtime();
    mandelbrot_pseudo_dynamic_parallelism(d_dwells, h_nextSize, d_nextSize, d_offsets1, d_offsets2, w, h, complex(-1.5, -1), complex(0.5, 1), W / INIT_SUBDIV, 1);
	double t2 = omp_get_wtime();

	cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
	gpu_time = t2 - t1;
	
	// save the image to PNG file
	save_image(IMAGE_PATH, h_dwells, w, h);

	// print performance
	//printf("Mandelbrot set computed in %.5lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);

    printf("%f\n", gpu_time);
	// free data
	cudaFree(d_dwells);
	free(h_dwells);
	return 0;
}  // main
