#pragma once

/** find the dwell for the pixel */
__device__ int pixel_dwell(unsigned int w, unsigned int h, complex cmin,
                           complex cmax, unsigned int x, unsigned int y,
                           unsigned int MAX_DWELL) {
    complex dc = cmax - cmin;
    float fx = (float)x / w, fy = (float)y / h;
    complex c = cmin + complex(fx * dc.re, fy * dc.im);
    int dwell = 0;
    complex z = c;
    while (dwell < MAX_DWELL && abs2(z) < 2 * 2) {
        z = z * z + c;
        dwell++;
    }
    return dwell;
} // pixel_dwell

/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
                                                                element, -1 =
   dwells are different */
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (0xffffffff)

__device__ int same_dwell(int d1, int d2, unsigned int MAX_DWELL) {
    if (d1 == d2)
        return d1;
    else if (d1 == NEUT_DWELL || d2 == NEUT_DWELL)
        return min(d1, d2);
    else
        return DIFF_DWELL;
} // same_dwell

/** gets the color, given the dwell (on host) */
#define CUT_DWELL (MAX_DWELL / 2)
void dwell_color(int *r, int *g, int *b, int dwell, unsigned int MAX_DWELL) {
    // black for the Mandelbrot set
    /*if (dwell == 667){
                                    *r = 156;
                                    *g = 39;
                                    *b = 176;
    } else*/
    if (dwell >= MAX_DWELL) {
        *r = 253;
        *g = 231;
        *b = 206;
    } else {
        // cut at zero
        if (dwell < 0)
            dwell = 0;
        if (dwell <= CUT_DWELL) {
            // from black to blue the first half
            double factor = (double)dwell / (double)CUT_DWELL;
            *r = 255 - dwell * 8 * 255 / (CUT_DWELL * 8);
            *b = 255;
            *g = 255 - dwell * 8 * 255 / (CUT_DWELL * 8);
        } else {
            // from blue to white for the second half
            double factor =
                (double)(dwell - CUT_DWELL) / (double)(MAX_DWELL - CUT_DWELL);
            *g = (dwell - CUT_DWELL) * 255 / (double)(MAX_DWELL - CUT_DWELL);
            *r = (dwell - CUT_DWELL) * 254 / (double)(MAX_DWELL - CUT_DWELL);
            *b = (dwell - CUT_DWELL) * (255 - (255 - 61)) /
                 (double)(MAX_DWELL - CUT_DWELL);
            255 - (255 - 61) * factor;
        }
    }
} // dwell_color

/** save the dwell into a PNG file
    @remarks: code
        to save PNG file taken from here (error handling is removed):
    http://www.labbookpages.co.uk/software/imgProc/libPNG.html
 */
void save_image(const char *filename, int *dwells, uint64_t w, uint64_t h,
                unsigned int MAX_DWELL) {
    png_bytep row;

    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // exception handling
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    // write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, w, h, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    // set title
    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    png_set_text(png_ptr, info_ptr, &title_text, 1);
    png_write_info(png_ptr, info_ptr);

    // write image data/

    row = (png_bytep)malloc(3 * w * sizeof(png_byte));
    for (uint64_t y = 0; y < h; y++) {
        for (uint64_t x = 0; x < w; x++) {
            int r, g, b;
            dwell_color(&r, &g, &b, dwells[y * w + x], MAX_DWELL);
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
} // save_image

int checkArray(int *a, int *b, unsigned int w, unsigned int h) {
    unsigned int cont = 0;
    for (unsigned int i = 0; i < w * h; i++) {
        if (a[i] != b[i]) {
            // printf("%i, %i\n", i%h, i/h);
            cont++;
        }
    }
    return cont;
}

/** a useful function to compute the number of threads */
__host__ __device__ int divup(uint64_t x, uint64_t y) {
    return x / y + (x % y ? 1 : 0);
}