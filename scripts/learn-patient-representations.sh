#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=ehr100k

indir=$datadir/$disease_dt

outdir=$datadir/experiments

sampling=10000

gpu=0

eval_baseline=''
# eval_baseline='--eval-baseline'

# without sampling
CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt $eval_baseline

# with sampling
# CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt -s $sampling $eval_baseline

