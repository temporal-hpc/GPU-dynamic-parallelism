NAME=mandelbrot
GRIDLINES=GRIDLINES
VERBOSE=no
SRC=main.cu
TGT=$(NAME)
TMP=*~ $(TGT)
BSX=32
BSY=32
REPEATS=1
ARCH=sm_75
build:	$(TGT)
$(TGT):	$(SRC) Makefile
	nvcc -O3 -arch=${ARCH} -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -D${VERBOSE} -D${GRIDLINES} -DBSY=$(BSY) -DBSX=$(BSX) -DREPEATS=$(REPEATS)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
