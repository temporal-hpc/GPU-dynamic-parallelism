NAME=mandelbrot-1ker
SRC=$(NAME).cu
TGT=$(NAME)
TMP=*~ $(TGT)

H=0
W=0
MAX_DWELL=0

MAX_DEPTH=5
MIN_SIZE=32

build:	$(TGT)
$(TGT):	$(SRC) makefile
	nvcc -O3 -arch=sm_75 -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -DW=$(W) -DH=$(H) -DMAX_DWELL=$(MAX_DWELL) -DMAX_DEPTH=$(MAX_DEPTH) -DMIN_SIZE=$(MIN_SIZE)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
