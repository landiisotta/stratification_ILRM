#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3

indir=$datadir/experiments/ehr-804370-test-1
outdir=$indir/encodings

gpu=1

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

../../myvenv/bin/python -u $projdir/src_v3/baselines.py $indir $outdir $test_set
