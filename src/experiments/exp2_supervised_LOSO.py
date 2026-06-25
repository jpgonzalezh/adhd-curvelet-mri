# ============================================================
# LOSO EXPERIMENTO 2 - OPCIÓN B
# Train: N-1 sitios → Test: sitio held-out
# Responde Comment 4: ¿las regiones identificadas en N-1 sitios
# predicen el sitio held-out?
# Inner CV sobre training para seleccionar mejor (región, param)
# Permutation test + FDR/Bonferroni
# Rank stability + Jaccard
# ============================================================

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import os
import time
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                             roc_auc_score, matthews_corrcoef,
                             average_precision_score)
from statsmodels.stats.multitest import multipletests
from collections import Counter

# ── 1. CONFIGURACIÓN GLOBAL ───────────────────────────────────
RANDOM_SEED = 70
N_BOOTSTRAP = 500
N_PERMUT    = 1000
N_JOBS      = -3
from config import DATA_ROOT as BASE_PATH  # set via DATA_ROOT env var

SITES_LOSO = ['KKI', 'NYU', 'NeuroIMAGE', 'OHSU', 'Peking']

configs = [
    {
        'csv':    f'{BASE_PATH}/comparison_sample/curvelet_data_adhd_201_loso.csv',
        'output': f'{BASE_PATH}/comparison_sample/results_loso_exp2',
        'name':   '201'
    },
    {
        'csv':    f'{BASE_PATH}/649_subjects/curvelet_data_adhd_649_loso.csv',
        'output': f'{BASE_PATH}/649_subjects/results_loso_exp2',
        'name':   '649'
    },
    {
        'csv':    f'{BASE_PATH}/875_subjects/curvelet_data_adhd_875_loso.csv',
        'output': f'{BASE_PATH}/875_subjects/results_loso_exp2',
        'name':   '875'
    }
]

# ── 2. FUNCIÓN BOOTSTRAP CI ───────────────────────────────────
def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=500, seed=70):
    rng = np.random.RandomState(seed)
    metrics_boot = {'Accuracy': [], 'BalancedAcc': [], 'ROC_AUC': [],
                    'PR_AUC': [], 'MCC': [], 'Sensitivity': [],
                    'Specificity': [], 'F1': []}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt  = y_true[idx]
        yp  = y_pred[idx]
        ypr = y_prob[idx]

        if len(np.unique(yt)) < 2:
            continue
        conf = confusion_matrix(yt, yp)
        if conf.shape != (2, 2):
            continue

        tn, fp, fn, tp = conf.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = (2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0)

        metrics_boot['Accuracy'].append(metrics.accuracy_score(yt, yp))
        metrics_boot['BalancedAcc'].append(balanced_accuracy_score(yt, yp))
        metrics_boot['ROC_AUC'].append(roc_auc_score(yt, ypr))
        metrics_boot['PR_AUC'].append(average_precision_score(yt, ypr))
        metrics_boot['MCC'].append(matthews_corrcoef(yt, yp))
        metrics_boot['Sensitivity'].append(sens)
        metrics_boot['Specificity'].append(spec)
        metrics_boot['F1'].append(f1)

    ci = {}
    for metric, values in metrics_boot.items():
        values = np.array(values)
        ci[f'{metric}_CI_low']  = np.percentile(values, 2.5)
        ci[f'{metric}_CI_high'] = np.percentile(values, 97.5)

    ba_samples = np.array(metrics_boot['BalancedAcc'])
    ci['p_value_bootstrap'] = (np.mean(ba_samples <= 0.5)
                               if len(ba_samples) > 0 else 1.0)
    return ci

# ── 3. INNER CV — seleccionar mejor (región, param) ──────────
def run_inner_cv(features_dict_train, labels_dict_train, RANDOM_SEED):
    inner_kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    best_inner_ba = -1
    best_region   = None
    best_param    = None

    for (region, param), X_all in features_dict_train.items():
        y_all = labels_dict_train[(region, param)]

        if len(np.unique(y_all)) < 2:
            continue

        inner_ba_scores = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_all):
            X_it = X_all[inner_train_idx]
            X_iv = X_all[inner_val_idx]
            y_it = y_all[inner_train_idx]
            y_iv = y_all[inner_val_idx]

            if len(np.unique(y_it)) < 2 or len(np.unique(y_iv)) < 2:
                continue

            scaler = StandardScaler()
            X_it_s = scaler.fit_transform(X_it)
            X_iv_s = scaler.transform(X_iv)

            clf = svm.SVC(kernel='linear', probability=True,
                          class_weight='balanced',
                          random_state=RANDOM_SEED)
            clf.fit(X_it_s, y_it)
            inner_ba_scores.append(
                balanced_accuracy_score(y_iv, clf.predict(X_iv_s)))

        if not inner_ba_scores:
            continue

        mean_ba = np.mean(inner_ba_scores)
        if mean_ba > best_inner_ba:
            best_inner_ba = mean_ba
            best_region   = region
            best_param    = param

    return best_region, best_param, best_inner_ba

# ── 4. FUNCIÓN POR SITIO HELD-OUT ─────────────────────────────
def run_loso_site(site_test, data, feature_cols,
                  parameters, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT):

    print(f"    → Site held-out: {site_test}")

    data_train = data[data['site_id'] != site_test].copy()
    data_test  = data[data['site_id'] == site_test].copy()

    if len(data_test) == 0:
        print(f"    ⚠️ Sin sujetos para {site_test} — omitiendo")
        return None
    if data_train['label_binary'].nunique() < 2:
        return None
    if data_test['label_binary'].nunique() < 2:
        print(f"    ⚠️ {site_test} no tiene ambas clases — omitiendo")
        return None

    n_regions = data_train['region_index'].nunique()
    n_train   = len(data_train) // n_regions
    n_test    = len(data_test)  // data_test['region_index'].nunique()
    print(f"    Training: {n_train} | Test: {n_test}")

    # Regiones válidas en training con ambas clases
    regions_all   = sorted(data_train['region_index'].unique())
    valid_regions = []
    for region in regions_all:
        rt  = data_train[data_train['region_index'] == region]
        rte = data_test[data_test['region_index'] == region]
        if rt['label_binary'].nunique() == 2 and len(rte) > 0 and \
           rte['label_binary'].nunique() == 2:
            valid_regions.append(region)

    if not valid_regions:
        return None

    sequence_alpha = list(range(0, 243, 3))
    sequence_beta  = list(range(1, 243, 3))
    sequence_mu    = list(range(2, 243, 3))

    features_dict_train = {}
    labels_dict_train   = {}
    features_dict_test  = {}
    labels_dict_test    = {}

    for param in parameters:
        seq = (sequence_alpha if param == 'alpha' else
               sequence_beta  if param == 'beta'  else sequence_mu)
        for region in valid_regions:
            rt = data_train[
                data_train['region_index'] == region
            ].reset_index(drop=True)
            features_dict_train[(region, param)] = rt[feature_cols].values[:, seq]
            labels_dict_train[(region, param)]   = rt['label_binary'].values

            rte = data_test[
                data_test['region_index'] == region
            ].reset_index(drop=True)
            features_dict_test[(region, param)]  = rte[feature_cols].values[:, seq]
            labels_dict_test[(region, param)]    = rte['label_binary'].values

    if not features_dict_train:
        return None

    # Inner CV sobre training → seleccionar mejor región
    best_region, best_param, best_inner_ba = run_inner_cv(
        features_dict_train, labels_dict_train, RANDOM_SEED)

    if best_region is None:
        return None

    X_train = features_dict_train[(best_region, best_param)]
    y_train = labels_dict_train[(best_region, best_param)]
    X_test  = features_dict_test.get((best_region, best_param))
    y_test  = labels_dict_test.get((best_region, best_param))

    if X_test is None or len(np.unique(y_test)) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf_final = svm.SVC(kernel='linear', probability=True,
                        class_weight='balanced',
                        random_state=RANDOM_SEED)
    clf_final.fit(X_train_s, y_train)

    y_pred = clf_final.predict(X_test_s)
    y_prob = clf_final.predict_proba(X_test_s)[:, 1]

    ba_real = balanced_accuracy_score(y_test, y_pred)
    ci = bootstrap_ci(y_test, y_pred, y_prob, N_BOOTSTRAP, RANDOM_SEED)

    conf = confusion_matrix(y_test, y_pred)
    if conf.shape != (2, 2):
        return None

    tn, fp, fn, tp = conf.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = (2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0)

    # ── PERMUTATION TEST ──────────────────────────────────────
    rng = np.random.RandomState(RANDOM_SEED)
    ba_permuted = []
    for _ in range(N_PERMUT):
        y_train_perm = rng.permutation(y_train)
        clf_p = svm.SVC(kernel='linear', probability=True,
                        class_weight='balanced',
                        random_state=RANDOM_SEED)
        clf_p.fit(X_train_s, y_train_perm)
        y_pred_p = clf_p.predict(X_test_s)
        if len(np.unique(y_test)) >= 2:
            ba_permuted.append(balanced_accuracy_score(y_test, y_pred_p))

    ba_permuted  = np.array(ba_permuted)
    p_value_perm = np.mean(ba_permuted >= ba_real)

    return {
        'Site_test':            site_test,
        'N_train':              n_train,
        'N_test':               n_test,
        'Best_region':          best_region,
        'Best_parameter':       best_param,
        'Best_inner_BA':        best_inner_ba,
        'Accuracy':             metrics.accuracy_score(y_test, y_pred),
        'BalancedAcc':          ba_real,
        'BalancedAcc_CI_low':   ci['BalancedAcc_CI_low'],
        'BalancedAcc_CI_high':  ci['BalancedAcc_CI_high'],
        'ROC_AUC':              roc_auc_score(y_test, y_prob),
        'ROC_AUC_CI_low':       ci['ROC_AUC_CI_low'],
        'ROC_AUC_CI_high':      ci['ROC_AUC_CI_high'],
        'PR_AUC':               average_precision_score(y_test, y_prob),
        'PR_AUC_CI_low':        ci['PR_AUC_CI_low'],
        'PR_AUC_CI_high':       ci['PR_AUC_CI_high'],
        'MCC':                  matthews_corrcoef(y_test, y_pred),
        'MCC_CI_low':           ci['MCC_CI_low'],
        'MCC_CI_high':          ci['MCC_CI_high'],
        'Sensitivity':          sens,
        'Sens_CI_low':          ci['Sensitivity_CI_low'],
        'Sens_CI_high':         ci['Sensitivity_CI_high'],
        'Specificity':          spec,
        'Spec_CI_low':          ci['Specificity_CI_low'],
        'Spec_CI_high':         ci['Specificity_CI_high'],
        'F1':                   f1,
        'F1_CI_low':            ci['F1_CI_low'],
        'F1_CI_high':           ci['F1_CI_high'],
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Conf_matrix':          f'[[{tn},{fp}],[{fn},{tp}]]',
        'p_value_permutation':  p_value_perm,
        'p_value_bootstrap':    ci['p_value_bootstrap'],
        'ba_null_mean':         ba_permuted.mean(),
        'ba_null_std':          ba_permuted.std()
    }

# ── 5. FUNCIÓN PRINCIPAL POR CONJUNTO ────────────────────────
def run_experiment(config, SITES_LOSO):

    name       = config['name']
    csv_path   = config['csv']
    output_dir = config['output']

    print(f"\n{'='*60}")
    print(f"LOSO EXP 2 OPCIÓN B — CONJUNTO: {name}")
    print(f"Train: N-1 sitios → Test: sitio held-out")
    print(f"{'='*60}")
    start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [f'curv_{i}' for i in range(1, 244)]
    data = pd.read_csv(csv_path, header=None, low_memory=False)
    data.columns = ['region_index', 'site_id', 'dx_group'] + feature_cols
    data['label_binary'] = data['dx_group'].apply(
        lambda x: 0 if x == 0 else 1)
    data = data[data['site_id'].isin(SITES_LOSO)]

    parameters      = ['alpha', 'beta', 'mu']
    sites_available = [s for s in SITES_LOSO
                       if s in data['site_id'].unique()]

    n_regions = data['region_index'].nunique()
    print(f"Sitios disponibles: {sites_available}")
    for site in sites_available:
        n = len(data[data['site_id'] == site]) // n_regions
        print(f"  {site}: {n} sujetos")

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_loso_site)(
            site_test, data, feature_cols,
            parameters, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT
        )
        for site_test in sites_available
    )

    results = [r for r in results if r is not None]

    if not results:
        print(f"⚠️ No hay resultados para {name}")
        return None

    df_results = pd.DataFrame(results)

    # ── FDR + BONFERRONI ──────────────────────────────────────
    p_values = df_results['p_value_permutation'].values
    rejected_fdr,  pvals_fdr,  _, _ = multipletests(
        p_values, alpha=0.05, method='fdr_bh')
    rejected_bonf, pvals_bonf, _, _ = multipletests(
        p_values, alpha=0.05, method='bonferroni')

    df_results['p_value_fdr']        = pvals_fdr
    df_results['significant_fdr']    = rejected_fdr
    df_results['p_value_bonferroni'] = pvals_bonf
    df_results['significant_bonf']   = rejected_bonf

    # ── RANK STABILITY ────────────────────────────────────────
    region_counts = Counter(zip(
        df_results['Best_region'], df_results['Best_parameter']))
    n_sites = len(sites_available)
    df_rank = pd.DataFrame([
        {'Region': r, 'Parameter': p,
         'Selection_count': c,
         'Selection_freq_%': round(c / n_sites * 100, 1)}
        for (r, p), c in region_counts.most_common()
    ])

    # ── GUARDAR ───────────────────────────────────────────────
    df_results.to_csv(
        f'{output_dir}/loso_results_{name}.csv', index=False)
    df_rank.to_csv(
        f'{output_dir}/loso_rank_stability_{name}.csv', index=False)

    df_confmat = df_results[[
        'Site_test', 'N_train', 'N_test',
        'Best_region', 'Best_parameter',
        'TN', 'FP', 'FN', 'TP', 'Conf_matrix',
        'Sensitivity', 'Specificity', 'Accuracy', 'BalancedAcc'
    ]].copy()
    df_confmat.to_csv(
        f'{output_dir}/loso_supplementary_confmat_{name}.csv', index=False)

    elapsed    = time.time() - start
    n_sig_fdr  = rejected_fdr.sum()
    n_sig_bonf = rejected_bonf.sum()

    print(f"\n✅ {name} completado en {elapsed:.1f} segundos")
    print(f"Sitios sig FDR:        {n_sig_fdr}/{n_sites}")
    print(f"Sitios sig Bonferroni: {n_sig_bonf}/{n_sites}")
    print(f"\n=== RESULTADOS LOSO ({name}) ===")
    print(df_results[[
        'Site_test', 'N_train', 'N_test',
        'Best_region', 'Best_parameter',
        'BalancedAcc', 'BalancedAcc_CI_low', 'BalancedAcc_CI_high',
        'ROC_AUC', 'MCC',
        'p_value_permutation', 'p_value_fdr', 'significant_fdr'
    ]].to_string())
    print(f"\n=== RANK STABILITY ===")
    print(df_rank.to_string())

    return set(zip(df_results['Best_region'], df_results['Best_parameter']))

# ── 6. MAIN + JACCARD ─────────────────────────────────────────
if __name__ == '__main__':
    total_start = time.time()

    rank_stability_all = {}
    for config in configs:
        top_regions = run_experiment(config, SITES_LOSO)
        if top_regions is not None:
            rank_stability_all[config['name']] = top_regions

    # ── JACCARD ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("JACCARD OVERLAP LOSO EXP 2 — ENTRE CONJUNTOS")
    print(f"{'='*60}")

    names = list(rank_stability_all.keys())
    jaccard_rows = []

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            set_a        = rank_stability_all[names[i]]
            set_b        = rank_stability_all[names[j]]
            intersection = len(set_a & set_b)
            union        = len(set_a | set_b)
            jaccard      = intersection / union if union > 0 else 0
            print(f"  {names[i]} vs {names[j]}: "
                  f"Jaccard = {jaccard:.3f} ({intersection}/{union})")
            jaccard_rows.append({
                'Set_A': names[i], 'Set_B': names[j],
                'Intersection': intersection,
                'Union': union, 'Jaccard': jaccard
            })

    df_jaccard = pd.DataFrame(jaccard_rows)
    for config in configs:
        df_jaccard.to_csv(
            f'{config["output"]}/loso_jaccard_exp2.csv', index=False)

    total_elapsed = time.time() - total_start
    print(f"\nTiempo total: {total_elapsed:.1f} segundos")