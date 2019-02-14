#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=multiple_myeloma

indir=$datadir/$disease_dt

outdir=$datadir/experiments

gpu=2

source ../myvenv/bin/activate

CUDA_VISIBLE_DEVICES=$gpu python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt

deactivate
