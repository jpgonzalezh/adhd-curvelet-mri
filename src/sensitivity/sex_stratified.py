# ============================================================
# SENSITIVITY ANALYSIS — SEX STRATIFIED (separado)
# Corre Exp 2 completo (nested CV) por separado en:
#   - Solo mujeres (N=333: 264 TDC + 69 ADHD)
#   - Solo hombres (N=541: 287 TDC + 254 ADHD)
# Mismo pipeline que exp2_5inner.py (inner=10, outer=10)
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

PHENO_PATH = f'{BASE_PATH}/ADHD200phenotypics.csv'
CURV_PATH  = f'{BASE_PATH}/875_subjects/curvelet_data_adhd_completed_axial_4scales.csv'

configs = [
    {'output': f'{BASE_PATH}/875_subjects/results_exp2_female_only', 'name': 'female', 'gender_filter': 'Female'},
    {'output': f'{BASE_PATH}/875_subjects/results_exp2_male_only',   'name': 'male',   'gender_filter': 'Male'}
]

# ── 2. OBTENER IDS POR SEXO ────────────────────────────────────
def get_ids_by_sex(gender_filter):
    pheno = pd.read_csv(PHENO_PATH)
    pheno['DX_clean']   = pd.to_numeric(pheno['DX'], errors='coerce')
    pheno['subject_id'] = pheno['ScanDir ID'].astype(str).str.strip()
    pheno['label']      = pheno['DX_clean'].apply(
        lambda x: 'TDC' if x==0 else ('ADHD' if x in [1,2,3] else None))
    pheno['Gender_name'] = pheno['Gender'].map({0:'Female', 1:'Male'})

    curv_ids = pd.read_csv(CURV_PATH, usecols=['subject_id'])
    curv_ids = curv_ids.dropna(subset=['subject_id'])
    curv_ids['subject_id'] = curv_ids['subject_id'].astype(str).str.strip().str.split('.').str[0]
    curv_ids = curv_ids.drop_duplicates(subset='subject_id')

    merged = curv_ids.merge(pheno[['subject_id','Gender_name','label']], on='subject_id', how='left')
    merged = merged.dropna(subset=['Gender_name','label'])
    sub = merged[merged['Gender_name']==gender_filter]

    print(f"  {gender_filter}: N={len(sub)}, TDC={sum(sub['label']=='TDC')}, ADHD={sum(sub['label']=='ADHD')}")
    return set(sub['subject_id'])

# ── 3. BOOTSTRAP CI ───────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=500, seed=70):
    rng = np.random.RandomState(seed)
    metrics_boot = {'Accuracy': [], 'BalancedAcc': [], 'ROC_AUC': [],
                    'PR_AUC': [], 'MCC': [], 'Sensitivity': [],
                    'Specificity': [], 'F1': []}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        conf = confusion_matrix(yt, yp)
        if conf.shape != (2, 2):
            continue
        tn, fp, fn, tp = conf.ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
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
    ci['p_value_bootstrap'] = np.mean(ba_samples <= 0.5) if len(ba_samples)>0 else 1.0
    return ci

# ── 4. INNER LOOP ──────────────────────────────────────────────
def run_inner_cv(train_idx, features_dict, labels_dict, RANDOM_SEED):
    inner_kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    best_inner_ba = -1
    best_region, best_param = None, None
    for (region, param), X_all in features_dict.items():
        y_all = labels_dict[(region, param)]
        X_train_outer = X_all[train_idx]
        y_train_outer = y_all[train_idx]
        if len(np.unique(y_train_outer)) < 2:
            continue
        inner_ba_scores = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train_outer):
            X_it, X_iv = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
            y_it, y_iv = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
            if len(np.unique(y_it))<2 or len(np.unique(y_iv))<2:
                continue
            scaler = StandardScaler()
            X_it_s = scaler.fit_transform(X_it)
            X_iv_s = scaler.transform(X_iv)
            clf = svm.SVC(kernel='linear', probability=True,
                          class_weight='balanced', random_state=RANDOM_SEED)
            clf.fit(X_it_s, y_it)
            inner_ba_scores.append(balanced_accuracy_score(y_iv, clf.predict(X_iv_s)))
        if not inner_ba_scores:
            continue
        mean_ba = np.mean(inner_ba_scores)
        if mean_ba > best_inner_ba:
            best_inner_ba, best_region, best_param = mean_ba, region, param
    return best_region, best_param, best_inner_ba

# ── 5. OUTER FOLD ──────────────────────────────────────────────
def run_outer_fold(outer_fold, train_idx, test_idx, features_dict,
                   labels_dict, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT):
    best_region, best_param, best_inner_ba = run_inner_cv(
        train_idx, features_dict, labels_dict, RANDOM_SEED)
    if best_region is None:
        return None
    X_best = features_dict[(best_region, best_param)]
    y_best = labels_dict[(best_region, best_param)]
    X_train, X_test = X_best[train_idx], X_best[test_idx]
    y_train, y_test = y_best[train_idx], y_best[test_idx]
    if len(np.unique(y_train))<2 or len(np.unique(y_test))<2:
        return None
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    clf = svm.SVC(kernel='linear', probability=True,
                  class_weight='balanced', random_state=RANDOM_SEED)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    ba_real = balanced_accuracy_score(y_test, y_pred)
    ci = bootstrap_ci(y_test, y_pred, y_prob, N_BOOTSTRAP, RANDOM_SEED)
    conf = confusion_matrix(y_test, y_pred)
    if conf.shape != (2,2):
        return None
    tn, fp, fn, tp = conf.ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
    rng = np.random.RandomState(RANDOM_SEED + outer_fold)
    ba_permuted = []
    for _ in range(N_PERMUT):
        y_perm = rng.permutation(y_train)
        clf_p = svm.SVC(kernel='linear', probability=True,
                        class_weight='balanced', random_state=RANDOM_SEED)
        clf_p.fit(X_train_s, y_perm)
        y_pred_p = clf_p.predict(X_test_s)
        if len(np.unique(y_test))>=2:
            ba_permuted.append(balanced_accuracy_score(y_test, y_pred_p))
    ba_permuted = np.array(ba_permuted)
    p_value_perm = np.mean(ba_permuted >= ba_real)
    return {
        'Outer_fold': outer_fold+1, 'Best_region': best_region,
        'Best_parameter': best_param, 'Best_inner_BA': best_inner_ba,
        'Accuracy': metrics.accuracy_score(y_test, y_pred),
        'BalancedAcc': ba_real,
        'BalancedAcc_CI_low': ci['BalancedAcc_CI_low'],
        'BalancedAcc_CI_high': ci['BalancedAcc_CI_high'],
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'ROC_AUC_CI_low': ci['ROC_AUC_CI_low'],
        'ROC_AUC_CI_high': ci['ROC_AUC_CI_high'],
        'PR_AUC': average_precision_score(y_test, y_prob),
        'PR_AUC_CI_low': ci['PR_AUC_CI_low'],
        'PR_AUC_CI_high': ci['PR_AUC_CI_high'],
        'MCC': matthews_corrcoef(y_test, y_pred),
        'MCC_CI_low': ci['MCC_CI_low'], 'MCC_CI_high': ci['MCC_CI_high'],
        'Sensitivity': sens, 'Sens_CI_low': ci['Sensitivity_CI_low'],
        'Sens_CI_high': ci['Sensitivity_CI_high'],
        'Specificity': spec, 'Spec_CI_low': ci['Specificity_CI_low'],
        'Spec_CI_high': ci['Specificity_CI_high'],
        'F1': f1, 'F1_CI_low': ci['F1_CI_low'], 'F1_CI_high': ci['F1_CI_high'],
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Conf_matrix': f'[[{tn},{fp}],[{fn},{tp}]]',
        'p_value_permutation': p_value_perm,
        'p_value_bootstrap': ci['p_value_bootstrap'],
        'ba_null_mean': ba_permuted.mean(), 'ba_null_std': ba_permuted.std()
    }

# ── 6. FUNCIÓN PRINCIPAL POR SEXO ─────────────────────────────
def run_experiment(config):
    name       = config['name']
    output_dir = config['output']
    gender     = config['gender_filter']

    print(f"\n{'='*60}")
    print(f"CORRIENDO EXP 2 ESTRATIFICADO — {name.upper()}")
    print(f"{'='*60}")
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("Obteniendo IDs...")
    selected_ids = get_ids_by_sex(gender)

    print("Cargando curvelet data...")
    feature_cols = [f'curv_{i}' for i in range(1, 244)]
    data = pd.read_csv(CURV_PATH, usecols=['region_index','subject_id','dx_group'] + feature_cols)
    data = data.dropna(subset=['subject_id'])
    data['subject_id'] = data['subject_id'].astype(str).str.strip().str.split('.').str[0]
    data = data[data['subject_id'].isin(selected_ids)]
    data['label_binary'] = data['dx_group'].apply(lambda x: 0 if x==0 else 1)

    print(f"Sujetos únicos: {data['subject_id'].nunique()}")

    # Filtrar solo regiones con el mismo N de sujetos (completas)
    region_counts = data.groupby('region_index')['subject_id'].nunique()
    n_target = int(region_counts.mode()[0])
    regiones_completas = region_counts[region_counts == n_target].index
    data = data[data['region_index'].isin(regiones_completas)]
    print(f"Regiones completas (N={n_target}): {len(regiones_completas)}")

    sequence_alpha = list(range(0, 243, 3))
    sequence_beta  = list(range(1, 243, 3))
    sequence_mu    = list(range(2, 243, 3))
    regions_list   = sorted(data['region_index'].unique())
    parameters     = ['alpha', 'beta', 'mu']

    features_dict, labels_dict = {}, {}
    for param in parameters:
        seq = (sequence_alpha if param=='alpha' else
               sequence_beta if param=='beta' else sequence_mu)
        for region in regions_list:
            rd = data[data['region_index']==region].reset_index(drop=True)
            if rd['label_binary'].nunique() < 2:
                continue
            features_dict[(region, param)] = rd[feature_cols].values[:, seq]
            labels_dict[(region, param)]   = rd['label_binary'].values

    if not labels_dict:
        print(f"⚠️  No hay combinaciones válidas para {name}")
        return

    first_key  = list(labels_dict.keys())[0]
    n_subjects = len(labels_dict[first_key])
    print(f"Combinaciones válidas: {len(features_dict)}")
    print(f"N sujetos (region-aligned): {n_subjects}")

    # Verificación de seguridad: todas las combinaciones deben tener el mismo N
    sizes = set(len(v) for v in labels_dict.values())
    assert len(sizes) == 1, f"ERROR: tamaños inconsistentes entre regiones: {sizes}"

    outer_kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_outer_fold)(
            outer_fold, train_idx, test_idx,
            features_dict, labels_dict, RANDOM_SEED, N_BOOTSTRAP, N_PERMUT)
        for outer_fold, (train_idx, test_idx)
        in enumerate(outer_kf.split(np.arange(n_subjects)))
    )

    results  = [r for r in results if r is not None]
    df_outer = pd.DataFrame(results)

    if df_outer.empty:
        print(f"⚠️  Sin resultados válidos para {name}")
        return

    region_counts = Counter(zip(df_outer['Best_region'], df_outer['Best_parameter']))
    df_rank = pd.DataFrame([
        {'Region': r, 'Parameter': p, 'Selection_count': c,
         'Selection_freq_%': round(c/10*100, 1)}
        for (r,p), c in region_counts.most_common()
    ])

    p_values = df_outer['p_value_permutation'].values
    rejected_fdr, pvals_fdr, _, _   = multipletests(p_values, alpha=0.05, method='fdr_bh')
    rejected_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    df_outer['p_value_fdr']        = pvals_fdr
    df_outer['significant_fdr']    = rejected_fdr
    df_outer['p_value_bonferroni'] = pvals_bonf
    df_outer['significant_bonf']   = rejected_bonf

    df_outer.to_csv(f'{output_dir}/nested_cv_results_{name}.csv', index=False)
    df_rank.to_csv(f'{output_dir}/rank_stability_{name}.csv', index=False)

    elapsed = time.time() - start
    print(f"\n✅ {name.upper()} completado en {elapsed:.1f}s")
    print(f"Sig FDR: {rejected_fdr.sum()}/10")
    print(df_outer[['Outer_fold','Best_region','Best_parameter','BalancedAcc',
                    'ROC_AUC','MCC','Sensitivity','Specificity',
                    'p_value_fdr','significant_fdr']].to_string())

# ── 7. MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    total_start = time.time()
    for config in configs:
        run_experiment(config)
    print(f"\nTiempo total: {time.time()-total_start:.1f}s")
