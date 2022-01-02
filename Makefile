EXEC=gpuDP
GRIDLINES=no
VERBOSE=no
DEBUG=no
SRC=src/main.cu
TGT=bin/$(EXEC)
TMP=*~ $(TGT)
BSX=32
BSY=32
BENCHMARK=no
REPEATS=32
REALIZATIONS=10
ARCH=sm_80
build:
	nvcc -O3 -arch=${ARCH} -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -D${VERBOSE} -D${GRIDLINES} -D${DEBUG} -D${BENCHMARK} -DBSY=$(BSY) -DBSX=$(BSX) -DREPEATS=$(REPEATS) -DREALIZATIONS=${REALIZATIONS}

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
