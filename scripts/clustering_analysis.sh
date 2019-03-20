#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=mixed

indir=$datadir/$disease_dt

expdir=$datadir/experiments/mixed-w2v-200

sampling=10000
exclude_oth=True

# without sampling
../myvenv/bin/python -u $projdir/src_v2/clustering_inspection.py $indir $expdir $disease_dt
