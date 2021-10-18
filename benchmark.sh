#!/bin/bash
GPUPROG=./bin/gpuDP
for approach in 0 1 2 3
do
    for size in {0..16..1}
    do
        for B in 1 2 4 8 16 32 64 128 256
        do
            for blockSize in "32 16"
            do
                set -- $k
                bsx=$1
                bsy=$2
                for MAX_DEPTH in 1000
                do
                    for g0 in 2 4 8 16 32 64 128
                    do
                        for SUBDIV in 2 4 8
                        do
                            make -B -DREPEATS=10
                            a=$(exec ${GPUPROG} $approach $((2**${size})) $((2**${size})) -1.5 0.5 -1 1 512 $B $g0 $SUBDIV $MAX_DEPTH none)
                            if [ $? -eq 0 ]
                            then
                                echo $a >> data/output.dat
                                echo $a
                            elif [ $? -eq 22 ]
                            then
                                echo "0,0,0,0,0,0,0,0,0,0" >> data/output.dat
                                echo "algo malo"
                            else
                                echo $?
                            fi
                        done
                    done
                done
            done
        done
    done
done
