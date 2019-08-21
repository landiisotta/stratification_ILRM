#! /bin/zsh

projdir=../..
datadir=$projdir/data_v3

indir=$datadir/experiments/ehr-1608741

encdir_test1=$datadir/experiments/ehr-804370-test-1/encodings
encdir_test2=$datadir/experiments/ehr-804371-test-2/encodings

# choose between SNOMED and CCS-SINGLE
#disease='T2D'
disease='PD'
#disease='AD'
#disease='MM'

../../myvenv/bin/python -u $projdir/src_v3/inner-cl-validation.py $datadir $indir $encdir_test1 $encdir_test2 $disease
