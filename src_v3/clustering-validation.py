    # evaluate clustering
    gt_file = os.path.join(indir, ut.dt_files['diseases'])
    gt_disease = clu.load_mrn_disease(gt_file)
    min_clu = 2
    max_clu = 10

    if eval_baseline:
        print('\nRunning clustering on the TF-IDF vectors')
        datafile = os.path.join(indir, ut.dt_files['ehr'])
        mrn_idx, svd_mtx = clu.svd_tfidf(datafile, vocab_size)
        gt_disease_raw = [gt_disease[m][0] for m in mrn_idx]
        clu.eval_hierarchical_clustering(
            svd_mtx, gt_disease_raw, min_clu, max_clu)

    print('\nRunning clustering on the encoded vectors')
    gt_disease_enc = [gt_disease[m][0] for m in mrn]
    clu.eval_hierarchical_clustering(
        encoded, gt_disease_enc, min_clu, max_clu, preproc=True)
