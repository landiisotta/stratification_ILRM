#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=multiple_myeloma

indir=$datadir/$disease_dt

outdir=$datadir/experiments

gpu=3

eval_baseline=True

CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt -b $eval_baseline
