#!/bin/sh
rm -rf plots/*
./genplots.sh work
./genplots.sh wrf
./genplots.sh time
./genplots.sh speedup

