#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=ehr100k

indir=$datadir/$disease_dt

expdir=$datadir/experiments/ehr100k-w2v-softplus

sampling=10000
exclude_oth=True

# without sampling
../myvenv/bin/python -u $projdir/src_v2/clustering_inspection_lstm.py $indir $expdir $disease_dt
