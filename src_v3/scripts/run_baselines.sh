#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3

indir=$datadir/experiments/ehr-100
outdir = $indir/encodings

gpu=1

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

CUDA_VISIBLE_DEVICES=$gpu ../../myvenv/bin/python -u $projdir/src_v3/baselines.py $indir $test_set
