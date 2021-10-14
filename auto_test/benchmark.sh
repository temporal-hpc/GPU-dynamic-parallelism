for approach in 2
do
    for size in {0..96..1}
    do
        for MIN_SIZE in 1 2 4 8 16 32
        do
            for blockSize in "32 16"
            do
                set -- $k
                bsx=$1
                bsy=$2
                for MAX_DEPTH in 5 7 1000
                do
                    for SUBDIV in 2 4 8
                    do
                        make -B
                        a=$(exec ./mandelbrot $approach $((1024+1024*$size)) $((1024+1024*$size)) -1.5 0.5 -1 1 512 $MIN_SIZE 8 $SUBDIV $MAX_DEPTH none)
                        if [ $? -eq 0 ]
                        then
                            echo $a >> data/output-rtx.dat
                            echo $a
                        elif [ $? -eq 22 ]
                        then
                            echo "0,0,0,0,0,0,0,0,0,0" >> data/output-rtx.dat
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
