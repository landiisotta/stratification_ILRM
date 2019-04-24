#! /bin/zsh

clear
projdir=..
datadir=$projdir/data
<<<<<<< HEAD
disease_dt=neurodev-disorder
=======
disease_dt=mixed
>>>>>>> 65ddc35f1ec7e1479e2ac3120a9bcf57a739d130

indir=$datadir/$disease_dt

outdir=$datadir/experiments

sampling=10000

<<<<<<< HEAD
gpu=1

emb_file=$datadir/embeddings/word2vec-pheno-embedding-100.emb

eval_baseline=''
# eval_baseline='--eval-baseline'

# without sampling
CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt $eval_baseline -e $emb_file

# with sampling
# CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt -s $sampling $eval_baseline

