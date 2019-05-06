#! /bin/zsh

clear
projdir=../..
datadir=$projdir/data_v3


outdir=$datadir/experiments
indir=ehr-49-test-2

gpu=1

emb_file=$datadir/embeddings/word2vec-pheno-embedding-100.emb

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

sampling=$2

if [ ! -z "$sampling"]
then
    sampling="-s $sampling"
fi

# without sampling
CUDA_VISIBLE_DEVICES=$gpu ../../myvenv/bin/python -u $projdir/src_v3/patient_representations.py $indir $outdir $test_set $sampling -e $emb_file

# with sampling
# CUDA_VISIBLE_DEVICES=$gpu ../myvenv/bin/python -u $projdir/src_v2/patient_representations.py $indir $outdir $disease_dt -s $sampling $eval_baseline

