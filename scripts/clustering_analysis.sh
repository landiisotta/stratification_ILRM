#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=ehr100k

indir=$datadir/$disease_dt

expdir=$datadir/experiments/ehr100k-2019-03-14-05-09-15-nobn-noact-norelu-10-l64

sampling=10000
exclude_oth=True

# without sampling
../myvenv/bin/python -u $projdir/src_v2/clustering_inspection.py $indir $expdir $disease_dt
