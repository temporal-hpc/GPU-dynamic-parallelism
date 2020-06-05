for i in {0..64..4}
do
    for j in 128 256 512
    do
        for k in "64 4" "16 16" "32 16" "32 32" 
        do
            set -- $k
            bsx=$1
            bsy=$2
            for l in 3 4 5
            do 
                for p in "2 1 2 4 8" "4 3 3 16 32" "8 7 4 64 128"
                do
                    set -- $p
                    subdiv=$1
                    ex=$2
                    ep=$3
                    e=$4
                    e2=$5
                    make -B H=$((1024+1024*$i)) W=$((1024+1024*$i)) MAX_DWELL=$j MAX_DEPTH=$l BSX=$bsx BSY=$bsy SUBDIV=$subdiv SUBDIV_ELEMS=$e SUBDIV_ELEMS2=$e2 SUBDIV_ELEMSP=$ep SUBDIV_ELEMSX=$ex
                    a=$(exec ./mandelbrot)
                    echo $a >> data/output.dat
                done
            done
        done
    done
done
