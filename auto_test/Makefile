NAME=mandelbrot
SRC=$(NAME).cu
TGT=$(NAME)
TMP=*~ $(TGT)
BSX=32
BSY=32
REPEATS=1
build:	$(TGT)
$(TGT):	$(SRC) Makefile
	nvcc -O3 -arch=sm_80 -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -DBSY=$(BSY) -DBSX=$(BSX) -DREPEATS=$(REPEATS)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
