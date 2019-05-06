#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3

indir=$datadir
outdir=$indir/experiments/

test_set=$1

if [ ! -z "$test_set" ]
then 
    test_set="--test_set $test_set"
fi

../../myvenv/bin/python -u $projdir/src_v3/data-preprocessing.py $indir $outdir $test_set 
