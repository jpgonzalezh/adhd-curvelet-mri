# ============================================================
# EXPERIMENTO 4 - POR CENTRO - CORREGIDO
# Reporta media ± SD de los 10 folds del KFold
# El 20% held-out se usa solo para permutation test
# z-score dentro del fold, class_weight='balanced'
# ============================================================

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import os
import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
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
N_JOBS      = -1
from config import DATA_ROOT as BASE_PATH  # set via DATA_ROOT env var

configs = [
    {'csv': f'{BASE_PATH}/KKI/curvelet_data_adhd_raw_4scales.csv',
     'output': f'{BASE_PATH}/KKI/results_exp4_final',
     'name': 'KKI'},
    {'csv': f'{BASE_PATH}/Neuro/curvelet_data_adhd_raw_4scales.csv',
     'output': f'{BASE_PATH}/Neuro/results_exp4_final',
     'name': 'Neuro'},
    {'csv': f'{BASE_PATH}/NYU/curvelet_data_adhd_raw_4scales.csv',
     'output': f'{BASE_PATH}/NYU/results_exp4_final',
     'name': 'NYU'},
    {'csv': f'{BASE_PATH}/OHSU/curvelet_data_adhd_raw_4scales.csv',
     'output': f'{BASE_PATH}/OHSU/results_exp4_final',
     'name': 'OHSU'},
    {'csv': f'{BASE_PATH}/Peking/curvelet_data_adhd_raw_4scales.csv',
     'output': f'{BASE_PATH}/Peking/results_exp4_final',
     'name': 'Peking'},
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

# ── 3. FUNCIÓN POR REGIÓN+PARÁMETRO ──────────────────────────
def process_region_param(region, param, data, feature_cols,
                          seq, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT):

    region_data = data[data['region_index'] == region]
    if region_data['label_binary'].nunique() < 2:
        return None

    X = region_data[feature_cols].values[:, seq]
    y = region_data['label_binary'].values

    # Split 80/20 — el 20% solo para permutation test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    if len(np.unique(y_train)) < 2:
        return None

    # ── KFOLD SOBRE EL 80% — métricas principales ─────────────
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics = {
        'Accuracy': [], 'BalancedAcc': [], 'ROC_AUC': [],
        'PR_AUC': [], 'MCC': [], 'Sensitivity': [],
        'Specificity': [], 'F1': []
    }
    fold_rows = []

    for k, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train[train_idx]
        X_fold_val   = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val   = y_train[val_idx]

        if len(np.unique(y_fold_train)) < 2:
            continue
        if len(np.unique(y_fold_val)) < 2:
            continue

        scaler = StandardScaler()
        X_fold_train_s = scaler.fit_transform(X_fold_train)
        X_fold_val_s   = scaler.transform(X_fold_val)

        clf_fold = svm.SVC(kernel='linear', probability=True,
                           class_weight='balanced',
                           random_state=RANDOM_SEED)
        clf_fold.fit(X_fold_train_s, y_fold_train)

        y_pred_val = clf_fold.predict(X_fold_val_s)
        y_prob_val = clf_fold.predict_proba(X_fold_val_s)[:, 1]

        conf = confusion_matrix(y_fold_val, y_pred_val)
        if conf.shape != (2, 2):
            continue

        tn, fp, fn, tp = conf.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = (2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0)
        roc  = roc_auc_score(y_fold_val, y_prob_val) if len(np.unique(y_fold_val)) == 2 else np.nan
        pr   = average_precision_score(y_fold_val, y_prob_val) if len(np.unique(y_fold_val)) == 2 else np.nan
        ba   = balanced_accuracy_score(y_fold_val, y_pred_val)
        acc  = metrics.accuracy_score(y_fold_val, y_pred_val)
        mcc  = matthews_corrcoef(y_fold_val, y_pred_val)

        fold_metrics['Accuracy'].append(acc)
        fold_metrics['BalancedAcc'].append(ba)
        fold_metrics['ROC_AUC'].append(roc)
        fold_metrics['PR_AUC'].append(pr)
        fold_metrics['MCC'].append(mcc)
        fold_metrics['Sensitivity'].append(sens)
        fold_metrics['Specificity'].append(spec)
        fold_metrics['F1'].append(f1)

        fold_rows.append({
            'Region': region, 'Parameter': param, 'Fold': k,
            'Accuracy': acc, 'BalancedAcc': ba,
            'ROC_AUC': roc, 'PR_AUC': pr, 'MCC': mcc,
            'Sensitivity': sens, 'Specificity': spec, 'F1': f1,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Conf_matrix': f'[[{tn},{fp}],[{fn},{tp}]]'
        })

    if not fold_metrics['BalancedAcc']:
        return None

    # ── MÉTRICAS PRINCIPALES: media ± SD de los 10 folds ──────
    n_folds = len(fold_metrics['BalancedAcc'])

    # Bootstrap CI sobre los fold scores agregados
    # Usamos el modelo final en el 20% para bootstrap CI
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled  = scaler_final.transform(X_test)

    clf_final = svm.SVC(kernel='linear', probability=True,
                        class_weight='balanced',
                        random_state=RANDOM_SEED)
    clf_final.fit(X_train_scaled, y_train)

    y_pred_test = clf_final.predict(X_test_scaled)
    y_prob_test = clf_final.predict_proba(X_test_scaled)[:, 1]

    if len(np.unique(y_test)) < 2:
        return None

    ci = bootstrap_ci(y_test, y_pred_test,
                      y_prob_test, N_BOOTSTRAP, RANDOM_SEED)

    # ── PERMUTATION TEST sobre el 20% ─────────────────────────
    ba_real = np.mean(fold_metrics['BalancedAcc'])
    rng = np.random.RandomState(RANDOM_SEED)
    ba_permuted = []

    for _ in range(N_PERMUT):
        y_train_perm = rng.permutation(y_train)
        clf_perm = svm.SVC(kernel='linear', probability=True,
                           class_weight='balanced',
                           random_state=RANDOM_SEED)
        clf_perm.fit(X_train_scaled, y_train_perm)
        y_pred_perm = clf_perm.predict(X_test_scaled)
        if len(np.unique(y_test)) >= 2:
            ba_permuted.append(balanced_accuracy_score(y_test, y_pred_perm))

    ba_permuted  = np.array(ba_permuted)
    p_value_perm = np.mean(ba_permuted >= ba_real)

    # Confusion matrix agregada de los folds
    tn_m = sum(r['TN'] for r in fold_rows)
    fp_m = sum(r['FP'] for r in fold_rows)
    fn_m = sum(r['FN'] for r in fold_rows)
    tp_m = sum(r['TP'] for r in fold_rows)

    return {
        'fold_rows': fold_rows,
        'boot_row': {
            'Region':              region,
            'Parameter':           param,
            'N_subjects':          len(y),
            'N_folds':             n_folds,
            # Media ± SD de los 10 folds — métricas principales
            'Accuracy':            np.mean(fold_metrics['Accuracy']),
            'Accuracy_SD':         np.std(fold_metrics['Accuracy']),
            'BalancedAcc':         np.mean(fold_metrics['BalancedAcc']),
            'BalancedAcc_SD':      np.std(fold_metrics['BalancedAcc']),
            'BalancedAcc_CI_low':  ci['BalancedAcc_CI_low'],
            'BalancedAcc_CI_high': ci['BalancedAcc_CI_high'],
            'ROC_AUC':             np.nanmean(fold_metrics['ROC_AUC']),
            'ROC_AUC_SD':          np.nanstd(fold_metrics['ROC_AUC']),
            'ROC_AUC_CI_low':      ci['ROC_AUC_CI_low'],
            'ROC_AUC_CI_high':     ci['ROC_AUC_CI_high'],
            'MCC':                 np.mean(fold_metrics['MCC']),
            'MCC_SD':              np.std(fold_metrics['MCC']),
            'MCC_CI_low':          ci['MCC_CI_low'],
            'MCC_CI_high':         ci['MCC_CI_high'],
            'Sensitivity':         np.mean(fold_metrics['Sensitivity']),
            'Sensitivity_SD':      np.std(fold_metrics['Sensitivity']),
            'Sens_CI_low':         ci['Sensitivity_CI_low'],
            'Sens_CI_high':        ci['Sensitivity_CI_high'],
            'Specificity':         np.mean(fold_metrics['Specificity']),
            'Specificity_SD':      np.std(fold_metrics['Specificity']),
            'Spec_CI_low':         ci['Specificity_CI_low'],
            'Spec_CI_high':        ci['Specificity_CI_high'],
            'F1':                  np.mean(fold_metrics['F1']),
            'F1_SD':               np.std(fold_metrics['F1']),
            'F1_CI_low':           ci['F1_CI_low'],
            'F1_CI_high':          ci['F1_CI_high'],
            'TN': tn_m, 'FP': fp_m, 'FN': fn_m, 'TP': tp_m,
            'Conf_matrix':         f'[[{tn_m},{fp_m}],[{fn_m},{tp_m}]]',
            'p_value_permutation': p_value_perm,
            'p_value_bootstrap':   ci['p_value_bootstrap'],
            'ba_null_mean':        ba_permuted.mean(),
            'ba_null_std':         ba_permuted.std()
        }
    }

# ── 4. FUNCIÓN PRINCIPAL POR CENTRO ──────────────────────────
def run_experiment(config):

    name       = config['name']
    csv_path   = config['csv']
    output_dir = config['output']

    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 4 — CENTRO: {name}")
    print(f"{'='*60}")
    start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path, header=None)
    feature_cols = [f'curv_{i}' for i in range(1, 244)]
    data.columns = ['region_index', 'dx_group'] + feature_cols
    data['label_binary'] = data['dx_group'].apply(
        lambda x: 0 if x == 0 else 1)

    if data['label_binary'].nunique() < 2:
        print(f"⚠️ {name} no tiene ambas clases — omitiendo")
        return None

    n_subjects = int(data.groupby('region_index').size().mode()[0])
    regions_complete = data.groupby('region_index').size()
    regions_complete = regions_complete[regions_complete == n_subjects].index
    data = data[data['region_index'].isin(regions_complete)]

    print(f"Sujetos: {n_subjects} | Regiones: {data['region_index'].nunique()}")
    print(f"Clases: {data['label_binary'].value_counts().to_dict()}")

    sequence_alpha = list(range(0, 243, 3))
    sequence_beta  = list(range(1, 243, 3))
    sequence_mu    = list(range(2, 243, 3))

    regions_list = sorted(data['region_index'].unique())
    parameters   = ['alpha', 'beta', 'mu']

    combinations = []
    for param in parameters:
        seq = (sequence_alpha if param == 'alpha' else
               sequence_beta  if param == 'beta'  else sequence_mu)
        for region in regions_list:
            combinations.append((region, param, seq))

    print(f"Combinaciones: {len(combinations)}")

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_region_param)(
            region, param, data, feature_cols,
            seq, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT
        )
        for region, param, seq in combinations
    )

    all_fold_rows = []
    all_boot_rows = []

    for r in results:
        if r is None:
            continue
        if r['fold_rows']:
            all_fold_rows.extend(r['fold_rows'])
        if r['boot_row']:
            all_boot_rows.append(r['boot_row'])

    df_per_fold  = pd.DataFrame(all_fold_rows)
    df_bootstrap = pd.DataFrame(all_boot_rows)

    # FDR + Bonferroni
    p_values = df_bootstrap['p_value_permutation'].values
    rejected_fdr,  pvals_fdr,  _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    rejected_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

    df_bootstrap['p_value_fdr']        = pvals_fdr
    df_bootstrap['significant_fdr']    = rejected_fdr
    df_bootstrap['p_value_bonferroni'] = pvals_bonf
    df_bootstrap['significant_bonf']   = rejected_bonf

    df_bootstrap = df_bootstrap.sort_values('BalancedAcc', ascending=False)

    # Guardar
    df_bootstrap.to_csv(f'{output_dir}/results_exp4_{name}.csv', index=False)
    df_per_fold[[
        'Region', 'Parameter', 'Fold',
        'TN', 'FP', 'FN', 'TP', 'Conf_matrix',
        'Sensitivity', 'Specificity', 'Accuracy', 'BalancedAcc'
    ]].to_csv(f'{output_dir}/supplementary_confmat_{name}.csv', index=False)

    elapsed   = time.time() - start
    n_sig_fdr = rejected_fdr.sum()

    print(f"\n✅ {name} completado en {elapsed:.1f} segundos")
    print(f"Regiones sig FDR: {n_sig_fdr}/{len(combinations)}")
    print(f"\n=== TOP 5 ({name}) ===")
    print(df_bootstrap[[
        'Region', 'Parameter', 'N_folds',
        'BalancedAcc', 'BalancedAcc_SD',
        'ROC_AUC', 'MCC',
        'p_value_permutation', 'p_value_fdr', 'significant_fdr'
    ]].head().to_string())

    return set(zip(df_bootstrap.head(10)['Region'],
                   df_bootstrap.head(10)['Parameter']))

# ── 5. MAIN + JACCARD ─────────────────────────────────────────
if __name__ == '__main__':
    total_start = time.time()

    rank_stability_all = {}
    for config in configs:
        top_regions = run_experiment(config)
        if top_regions is not None:
            rank_stability_all[config['name']] = top_regions

    print(f"\n{'='*60}")
    print("JACCARD OVERLAP ENTRE CENTROS — TOP 10 REGIONES")
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
            print(f"  {names[i]} vs {names[j]}: Jaccard = {jaccard:.3f} ({intersection}/{union})")
            jaccard_rows.append({'Site_A': names[i], 'Site_B': names[j],
                                  'Intersection': intersection,
                                  'Union': union, 'Jaccard': jaccard})

    pd.DataFrame(jaccard_rows).to_csv(
        f'{BASE_PATH}/jaccard_overlap_exp4.csv', index=False)

    print(f"\nTiempo total: {time.time()-total_start:.1f} segundos")