#!/bin/sh
rm -rf plots/*
./theo-genplots.sh work      optimal
./theo-genplots.sh wrf       optimal
./theo-genplots.sh time      optimal
./theo-genplots.sh speedup   optimal

