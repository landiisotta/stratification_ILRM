#! /bin/zsh

projdir = ../../
datadir = $projdir/data_v3

indir = $datadir/experiments/ehr-

outdir = $datadir/encodings

../../myvenv/bin/python -u $projdir/src_v3/clustering-validation.py $datadir $indir $outdir
