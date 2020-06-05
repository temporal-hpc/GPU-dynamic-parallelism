NAME=mandelbrot
SRC=$(NAME).cu
TGT=$(NAME)
TMP=*~ $(TGT)

H=0
W=0
MAX_DWELL=0

build:	$(TGT)
$(TGT):	$(SRC) Makefile
	nvcc -O3 -arch=sm_75 -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -DH=$(H) -DW=$(W) -DMAX_DWELL=$(MAX_DWELL)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
