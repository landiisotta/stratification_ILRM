#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3

indir=$datadir
outdir=$indir/experiments/

test_set='test-2.csv'

../../myvenv/bin/python -u $projdir/src_v3/data-preprocessing.py $indir $outdir $test_set 
