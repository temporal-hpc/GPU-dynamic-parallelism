NAME=mandelbrot
SRC=$(NAME).cu
TGT=$(NAME)
TMP=*~ $(TGT)

H=0
W=0
MAX_DWELL=0
BSX=0
BSY=0
MAX_DEPTH=5
MIN_SIZE=0
SUBDIV=0
SUBDIV_ELEMSX=0
SUBDIV_ELEMSP=0
SUBDIV_ELEMS=0
SUBDIV_ELEMS2=0


build:	$(TGT)
$(TGT):	$(SRC) Makefile
	nvcc -O3 -arch=sm_75 -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -DBSY=$(BSY) -DBSX=$(BSX)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
