#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
disease_dt=mixed

indir=$datadir/$disease_dt

outdir=$datadir/experiments

sampling=10000

gpu=1

emb_file=$datadir/embeddings/word2vec-pheno-embedding-100.emb

eval_baseline=''
# eval_baseline='--eval-baseline'

# without sampling
CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt $eval_baseline -e $emb_file

# with sampling
# CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt -s $sampling $eval_baseline

