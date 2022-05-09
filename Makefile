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
DP_RDC=-rdc=true -DRDC_TRUE
DP_PENDING_KERNEL_BUFFER=$$((1024*512))
build:
	nvcc -O3 -arch=${ARCH} ${DP_RDC} -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT) -D${VERBOSE} -D${GRIDLINES} -D${DEBUG} -D${BENCHMARK} -DBSX=$(BSX) -DBSY=$(BSY) -DREALIZATIONS=${REALIZATIONS} -DREPEATS=$(REPEATS) -DDP_PENDING_KERNEL_BUFFER=${DP_PENDING_KERNEL_BUFFER}

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
