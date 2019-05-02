#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3

indir=$datadir
outdir=$indir/experiments/

../../myvenv/bin/python -u $projdir/src_v3/data-preprocessing.py $indir $outdir --exclude-terms
