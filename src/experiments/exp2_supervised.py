# ============================================================
# EXPERIMENTO 2 - SUPERVISADO - COMPLETO FINAL
# Nested CV + Permutation test + FDR/Bonferroni
# Rank stability + Jaccard overlap
# Matrices de confusión (Supplementary)
# CSV crudo, z-score dentro del fold
# class_weight='balanced', bootstrap CI
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

configs = [
    {
        'csv':    f'{BASE_PATH}/comparison_sample/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/comparison_sample/results_exp2_final',
        'name':   '201'
    },
    {
        'csv':    f'{BASE_PATH}/649_subjects/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/649_subjects/results_exp2_final',
        'name':   '649'
    },
    {
        'csv':    f'{BASE_PATH}/875_subjects/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/875_subjects/results_exp2_final',
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
        f1   = (2 * prec * sens / (prec + sens)
                if (prec + sens) > 0 else 0)

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

# ── 3. FUNCIÓN INNER LOOP ─────────────────────────────────────
def run_inner_cv(train_idx, features_dict, labels_dict, RANDOM_SEED):

    inner_kf = KFold(n_splits=10, shuffle=True,
                     random_state=RANDOM_SEED)

    best_inner_ba = -1
    best_region   = None
    best_param    = None

    for (region, param), X_all in features_dict.items():
        y_all         = labels_dict[(region, param)]
        X_train_outer = X_all[train_idx]
        y_train_outer = y_all[train_idx]

        if len(np.unique(y_train_outer)) < 2:
            continue

        inner_ba_scores = []

        for inner_train_idx, inner_val_idx in inner_kf.split(
                X_train_outer):

            X_inner_train = X_train_outer[inner_train_idx]
            X_inner_val   = X_train_outer[inner_val_idx]
            y_inner_train = y_train_outer[inner_train_idx]
            y_inner_val   = y_train_outer[inner_val_idx]

            if len(np.unique(y_inner_train)) < 2:
                continue
            if len(np.unique(y_inner_val)) < 2:
                continue

            scaler = StandardScaler()
            X_inner_train_s = scaler.fit_transform(X_inner_train)
            X_inner_val_s   = scaler.transform(X_inner_val)

            clf = svm.SVC(kernel='linear', probability=True,
                          class_weight='balanced',
                          random_state=RANDOM_SEED)
            clf.fit(X_inner_train_s, y_inner_train)

            y_pred_inner = clf.predict(X_inner_val_s)
            ba = balanced_accuracy_score(y_inner_val, y_pred_inner)
            inner_ba_scores.append(ba)

        if len(inner_ba_scores) == 0:
            continue

        mean_inner_ba = np.mean(inner_ba_scores)
        if mean_inner_ba > best_inner_ba:
            best_inner_ba = mean_inner_ba
            best_region   = region
            best_param    = param

    return best_region, best_param, best_inner_ba

# ── 4. FUNCIÓN OUTER FOLD ─────────────────────────────────────
def run_outer_fold(outer_fold, train_idx, test_idx,
                   features_dict, labels_dict,
                   RANDOM_SEED, N_BOOTSTRAP, N_PERMUT):

    best_region, best_param, best_inner_ba = run_inner_cv(
        train_idx, features_dict, labels_dict, RANDOM_SEED)

    if best_region is None:
        return None

    X_best = features_dict[(best_region, best_param)]
    y_best = labels_dict[(best_region, best_param)]

    X_train_final = X_best[train_idx]
    X_test_final  = X_best[test_idx]
    y_train_final = y_best[train_idx]
    y_test_final  = y_best[test_idx]

    if len(np.unique(y_train_final)) < 2:
        return None
    if len(np.unique(y_test_final)) < 2:
        return None

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_final)
    X_test_scaled  = scaler_final.transform(X_test_final)

    clf_final = svm.SVC(kernel='linear', probability=True,
                        class_weight='balanced',
                        random_state=RANDOM_SEED)
    clf_final.fit(X_train_scaled, y_train_final)

    y_pred_test = clf_final.predict(X_test_scaled)
    y_prob_test = clf_final.predict_proba(X_test_scaled)[:, 1]

    ba_real = balanced_accuracy_score(y_test_final, y_pred_test)

    ci = bootstrap_ci(y_test_final, y_pred_test,
                      y_prob_test, N_BOOTSTRAP, RANDOM_SEED)

    conf = confusion_matrix(y_test_final, y_pred_test)
    if conf.shape != (2, 2):
        return None

    tn, fp, fn, tp = conf.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = (2 * prec * sens / (prec + sens)
            if (prec + sens) > 0 else 0)

    # ── PERMUTATION TEST ──────────────────────────────────────
    rng = np.random.RandomState(RANDOM_SEED + outer_fold)
    ba_permuted = []

    for _ in range(N_PERMUT):
        y_train_perm = rng.permutation(y_train_final)

        clf_perm = svm.SVC(kernel='linear', probability=True,
                           class_weight='balanced',
                           random_state=RANDOM_SEED)
        clf_perm.fit(X_train_scaled, y_train_perm)

        y_pred_perm = clf_perm.predict(X_test_scaled)

        if len(np.unique(y_test_final)) < 2:
            continue

        ba_perm = balanced_accuracy_score(y_test_final, y_pred_perm)
        ba_permuted.append(ba_perm)

    ba_permuted  = np.array(ba_permuted)
    p_value_perm = np.mean(ba_permuted >= ba_real)

    return {
        'Outer_fold': outer_fold + 1,
        'Best_region': best_region,
        'Best_parameter': best_param,
        'Best_inner_BA': best_inner_ba,
        'Accuracy': metrics.accuracy_score(y_test_final, y_pred_test),
        'BalancedAcc': ba_real,
        'BalancedAcc_CI_low': ci['BalancedAcc_CI_low'],
        'BalancedAcc_CI_high': ci['BalancedAcc_CI_high'],
        'ROC_AUC': roc_auc_score(y_test_final, y_prob_test),
        'ROC_AUC_CI_low': ci['ROC_AUC_CI_low'],
        'ROC_AUC_CI_high': ci['ROC_AUC_CI_high'],
        'PR_AUC': average_precision_score(y_test_final, y_prob_test),
        'PR_AUC_CI_low': ci['PR_AUC_CI_low'],
        'PR_AUC_CI_high': ci['PR_AUC_CI_high'],
        'MCC': matthews_corrcoef(y_test_final, y_pred_test),
        'MCC_CI_low': ci['MCC_CI_low'],
        'MCC_CI_high': ci['MCC_CI_high'],
        'Sensitivity': sens,
        'Sens_CI_low': ci['Sensitivity_CI_low'],
        'Sens_CI_high': ci['Sensitivity_CI_high'],
        'Specificity': spec,
        'Spec_CI_low': ci['Specificity_CI_low'],
        'Spec_CI_high': ci['Specificity_CI_high'],
        'F1': f1,
        'F1_CI_low': ci['F1_CI_low'],
        'F1_CI_high': ci['F1_CI_high'],
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Conf_matrix': f'[[{tn},{fp}],[{fn},{tp}]]',
        'p_value_permutation': p_value_perm,
        'p_value_bootstrap': ci['p_value_bootstrap'],
        'ba_null_mean': ba_permuted.mean(),
        'ba_null_std': ba_permuted.std()
    }

# ── 5. CALCULAR REGIONES COMUNES ──────────────────────────────
print("Calculando regiones comunes a los tres conjuntos...")

regiones_por_conjunto = {}

for config in configs:
    data_temp = pd.read_csv(config['csv'], header=None)
    feature_cols_temp = [f'curv_{i}' for i in range(1, 244)]
    data_temp.columns = ['region_index', 'dx_group'] + feature_cols_temp

    n_subjects = int(data_temp.groupby('region_index').size().mode()[0])
    completas = set(data_temp.groupby('region_index').size()[
        data_temp.groupby('region_index').size() == n_subjects].index)
    regiones_por_conjunto[config['name']] = completas
    print(f"  {config['name']} sujetos: {len(completas)} regiones completas")

REGIONES_COMUNES = sorted(
    regiones_por_conjunto['201'] &
    regiones_por_conjunto['649'] &
    regiones_por_conjunto['875']
)

print(f"\nRegiones comunes: {len(REGIONES_COMUNES)}")

# ── 6. FUNCIÓN PRINCIPAL POR CONJUNTO ────────────────────────
def run_experiment(config, REGIONES_COMUNES):

    name       = config['name']
    csv_path   = config['csv']
    output_dir = config['output']

    print(f"\n{'='*60}")
    print(f"CORRIENDO EXPERIMENTO 2 - CONJUNTO: {name} sujetos")
    print(f"{'='*60}")
    start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path, header=None)
    feature_cols = [f'curv_{i}' for i in range(1, 244)]
    data.columns = ['region_index', 'dx_group'] + feature_cols
    data['label_binary'] = data['dx_group'].apply(
        lambda x: 0 if x == 0 else 1)

    data = data[data['region_index'].isin(REGIONES_COMUNES)]

    print(f"Regiones usadas: {data['region_index'].nunique()}")
    print(f"Distribución de clases:")
    print(data['label_binary'].value_counts())

    sequence_alpha = list(range(0, 243, 3))
    sequence_beta  = list(range(1, 243, 3))
    sequence_mu    = list(range(2, 243, 3))

    regions_list = sorted(data['region_index'].unique())
    parameters   = ['alpha', 'beta', 'mu']

    print("Construyendo diccionario de features...")
    features_dict = {}
    labels_dict   = {}

    for param in parameters:
        seq = (sequence_alpha if param == 'alpha' else
               sequence_beta  if param == 'beta'  else sequence_mu)
        for region in regions_list:
            region_data = data[
                data['region_index'] == region
            ].reset_index(drop=True)

            if region_data['label_binary'].nunique() < 2:
                continue

            features_dict[(region, param)] = \
                region_data[feature_cols].values[:, seq]
            labels_dict[(region, param)] = \
                region_data['label_binary'].values

    first_key  = list(labels_dict.keys())[0]
    y_global   = labels_dict[first_key]
    n_subjects = len(y_global)

    print(f"Combinaciones válidas: {len(features_dict)}")
    print(f"Sujetos totales: {n_subjects}")
    print(f"Permutaciones por outer fold: {N_PERMUT}")

    outer_kf = KFold(n_splits=10, shuffle=True,
                     random_state=RANDOM_SEED)

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_outer_fold)(
            outer_fold, train_idx, test_idx,
            features_dict, labels_dict,
            RANDOM_SEED, N_BOOTSTRAP, N_PERMUT
        )
        for outer_fold, (train_idx, test_idx)
        in enumerate(outer_kf.split(np.arange(n_subjects)))
    )

    results  = [r for r in results if r is not None]
    df_outer = pd.DataFrame(results)

    # ── RANK STABILITY ────────────────────────────────────────
    region_counts = Counter(zip(
        df_outer['Best_region'],
        df_outer['Best_parameter']
    ))

    df_rank = pd.DataFrame([
        {'Region': r, 'Parameter': p,
         'Selection_count': c,
         'Selection_freq_%': round(c / 10 * 100, 1)}
        for (r, p), c in region_counts.most_common()
    ])

    # ── FDR + BONFERRONI ──────────────────────────────────────
    p_values = df_outer['p_value_permutation'].values

    rejected_fdr, pvals_fdr, _, _ = multipletests(
        p_values, alpha=0.05, method='fdr_bh')
    rejected_bonf, pvals_bonf, _, _ = multipletests(
        p_values, alpha=0.05, method='bonferroni')

    df_outer['p_value_fdr']        = pvals_fdr
    df_outer['significant_fdr']    = rejected_fdr
    df_outer['p_value_bonferroni'] = pvals_bonf
    df_outer['significant_bonf']   = rejected_bonf

    # ── GUARDAR EN CARPETA DEL CONJUNTO ──────────────────────
    df_outer.to_csv(
        f'{output_dir}/nested_cv_results_{name}.csv', index=False)
    df_rank.to_csv(
        f'{output_dir}/rank_stability_{name}.csv', index=False)

    # Suplementario
    df_confmat = df_outer[[
        'Outer_fold', 'Best_region', 'Best_parameter',
        'TN', 'FP', 'FN', 'TP', 'Conf_matrix',
        'Sensitivity', 'Specificity', 'Accuracy', 'BalancedAcc'
    ]].copy()
    df_confmat.to_csv(
        f'{output_dir}/supplementary_confmat_{name}.csv', index=False)

    elapsed = time.time() - start
    n_sig_fdr  = rejected_fdr.sum()
    n_sig_bonf = rejected_bonf.sum()

    print(f"\n✅ {name} sujetos completado en {elapsed:.1f} segundos")
    print(f"Folds significativos FDR:        {n_sig_fdr}/10")
    print(f"Folds significativos Bonferroni: {n_sig_bonf}/10")
    print(f"\n=== RANK STABILITY — TOP 5 ===")
    print(df_rank.head())
    print(f"\n=== RESULTADOS POR OUTER FOLD ===")
    print(df_outer[['Outer_fold', 'Best_region', 'Best_parameter',
                    'BalancedAcc', 'Conf_matrix',
                    'p_value_permutation',
                    'significant_fdr',
                    'significant_bonf']].to_string())

    return df_rank

# ── 7. CORRER LOS TRES CONJUNTOS + JACCARD ────────────────────
if __name__ == '__main__':
    total_start = time.time()

    rank_stability_all = {}
    all_rank_dfs = {}

    for config in configs:
        df_rank = run_experiment(config, REGIONES_COMUNES)
        if df_rank is not None:
            rank_stability_all[config['name']] = set(
                zip(df_rank['Region'], df_rank['Parameter']))
            all_rank_dfs[config['name']] = config['output']

    # ── JACCARD OVERLAP ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("JACCARD OVERLAP ENTRE CONJUNTOS")
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
                  f"Jaccard = {jaccard:.3f} "
                  f"({intersection}/{union} regiones comunes)")

            jaccard_rows.append({
                'Set_A': names[i],
                'Set_B': names[j],
                'Intersection': intersection,
                'Union': union,
                'Jaccard': jaccard
            })

    # ── GUARDAR JACCARD EN CADA CARPETA ──────────────────────
    df_jaccard = pd.DataFrame(jaccard_rows)
    for config in configs:
        df_jaccard.to_csv(
            f'{config["output"]}/jaccard_overlap_exp2.csv', index=False)
        print(f"  Jaccard guardado en: {config['output']}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"Tiempo total: {total_elapsed:.1f} segundos")
    print(f"{'='*60}")