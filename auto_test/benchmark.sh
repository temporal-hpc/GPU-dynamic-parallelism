for i in {0..77..1}
do
    for j in 32
    do
        for k in "32 32"
        do
            set -- $k
            bsx=$1
            bsy=$2
            for l in 5 7
            do
                for p in "2 1 2 4 8" "4 3 3 16 32" "8 7 4 64 128"
                do
                    set -- $p
                    subdiv=$1
                    ex=$2
                    ep=$3
                    e=$4
                    e2=$5
                    make -B H=$((1024+1024*$i)) W=$((1024+1024*$i)) MAX_DWELL=512 MIN_SIZE=$j  MAX_DEPTH=$l BSX=$bsx BSY=$bsy SUBDIV=$subdiv SUBDIV_ELEMS=$e SUBDIV_ELEMS2=$e2 SUBDIV_ELEMSP=$ep SUBDIV_ELEMSX=$ex
                    a=$(exec ./mandelbrot)
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
                      echo "ERROR!!!!"
                      exit 2
                    fi
                done
            done
        done
    done
done
