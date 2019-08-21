#! /bin/zsh

projdir=../..
datadir=$projdir/data_v3

indir=$datadir/experiments/ehr-804371-test-2

encdir=$indir/encodings

# choose between SNOMED and CCS-SINGLE
code='snomed'

../../myvenv/bin/python -u $projdir/src_v3/clustering-validation.py $datadir $indir $encdir $code
