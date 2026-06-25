# ============================================================
# EXPERIMENTO 1 - NO SUPERVISADO CORREGIDO - FINAL
# CSV crudo, z-score dentro del split
# K-means solo sobre training
# class_weight='balanced'
# Bootstrap CI + p-value desde bootstrap
# FDR + Bonferroni sobre p-values bootstrap
# SIN permutation test (no aplica a no supervisado)
# SIN nested CV (no aplica a no supervisado)
# ============================================================

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import os
import time
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                             roc_auc_score, matthews_corrcoef,
                             average_precision_score)
from statsmodels.stats.multitest import multipletests

# ── 1. CONFIGURACIÓN GLOBAL ───────────────────────────────────
RANDOM_SEED = 70
N_BOOTSTRAP = 500
N_JOBS      = -1
from config import DATA_ROOT as BASE_PATH  # set via DATA_ROOT env var

configs = [
    {
        'csv':    f'{BASE_PATH}/comparison_sample/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/comparison_sample/results_exp1_final',
        'name':   '201'
    },
    {
        'csv':    f'{BASE_PATH}/649_subjects/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/649_subjects/results_exp1_final',
        'name':   '649'
    },
    {
        'csv':    f'{BASE_PATH}/875_subjects/curvelet_data_adhd_raw_4scales.csv',
        'output': f'{BASE_PATH}/875_subjects/results_exp1_final',
        'name':   '875'
    }
]

# ── 2. FUNCIÓN BOOTSTRAP CI + P-VALUE ────────────────────────
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

    # P-value desde bootstrap
    # H0: BA = 0.5 (azar)
    # p-value = proporción de muestras donde BA <= 0.5
    ba_samples = np.array(metrics_boot['BalancedAcc'])
    ci['p_value_bootstrap'] = (np.mean(ba_samples <= 0.5)
                               if len(ba_samples) > 0 else 1.0)

    return ci

# ── 3. FUNCIÓN ALINEAR LABELS KMEANS ─────────────────────────
def align_kmeans_labels(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    acc_original = accuracy_score(y_true, y_pred)
    acc_flipped  = accuracy_score(y_true, 1 - y_pred)
    if acc_flipped > acc_original:
        return 1 - y_pred
    return y_pred

# ── 4. FUNCIÓN POR REGIÓN+PARÁMETRO ──────────────────────────
def process_region_param(region, param, data, feature_cols,
                          seq, RANDOM_SEED, N_BOOTSTRAP):

    region_data = data[data['region_index'] == region]
    if region_data['label_binary'].nunique() < 2:
        return None

    X = region_data[feature_cols].values[:, seq]
    y = region_data['label_binary'].values

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Z-score una sola vez
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # K-means solo sobre training
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(X_train_scaled)

    cluster_labels = kmeans.labels_
    cluster_labels = align_kmeans_labels(y_train, cluster_labels)

    # SVM con class weight
    clf = svm.SVC(kernel='linear', probability=True,
                  class_weight='balanced',
                  random_state=RANDOM_SEED)
    clf.fit(X_train_scaled, cluster_labels)

    # Predicción sobre test
    y_pred_test = clf.predict(X_test_scaled)
    y_prob_test = clf.predict_proba(X_test_scaled)[:, 1]

    if len(np.unique(y_test)) < 2:
        return None

    # Métricas puntuales
    conf_test = confusion_matrix(y_test, y_pred_test)
    if conf_test.shape != (2, 2):
        return None

    tn, fp, fn, tp = conf_test.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = (2 * prec * sens / (prec + sens)
            if (prec + sens) > 0 else 0)

    ba_real = balanced_accuracy_score(y_test, y_pred_test)

    # Bootstrap CI + p-value
    ci = bootstrap_ci(y_test, y_pred_test,
                      y_prob_test, N_BOOTSTRAP, RANDOM_SEED)

    return {
        'Region': region, 'Parameter': param,
        'N_test': len(y_test),
        'Accuracy': metrics.accuracy_score(y_test, y_pred_test),
        'BalancedAcc': ba_real,
        'BalancedAcc_CI_low': ci['BalancedAcc_CI_low'],
        'BalancedAcc_CI_high': ci['BalancedAcc_CI_high'],
        'ROC_AUC': roc_auc_score(y_test, y_prob_test),
        'ROC_AUC_CI_low': ci['ROC_AUC_CI_low'],
        'ROC_AUC_CI_high': ci['ROC_AUC_CI_high'],
        'PR_AUC': average_precision_score(y_test, y_prob_test),
        'PR_AUC_CI_low': ci['PR_AUC_CI_low'],
        'PR_AUC_CI_high': ci['PR_AUC_CI_high'],
        'MCC': matthews_corrcoef(y_test, y_pred_test),
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
        'p_value_bootstrap': ci['p_value_bootstrap']
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
    print(f"CORRIENDO EXPERIMENTO 1 - CONJUNTO: {name} sujetos")
    print(f"{'='*60}")
    start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # Cargar CSV
    data = pd.read_csv(csv_path, header=None)
    feature_cols = [f'curv_{i}' for i in range(1, 244)]
    data.columns = ['region_index', 'dx_group'] + feature_cols
    data['label_binary'] = data['dx_group'].apply(
        lambda x: 0 if x == 0 else 1)

    # Filtro regiones comunes
    data = data[data['region_index'].isin(REGIONES_COMUNES)]

    print(f"Regiones usadas: {data['region_index'].nunique()}")
    print(f"Distribución de clases:")
    print(data['label_binary'].value_counts())

    # Índices por parámetro
    sequence_alpha = list(range(0, 243, 3))
    sequence_beta  = list(range(1, 243, 3))
    sequence_mu    = list(range(2, 243, 3))

    regions_list = sorted(data['region_index'].unique())
    parameters   = ['alpha', 'beta', 'mu']

    # Crear combinaciones
    combinations = []
    for param in parameters:
        seq = (sequence_alpha if param == 'alpha' else
               sequence_beta  if param == 'beta'  else sequence_mu)
        for region in regions_list:
            combinations.append((region, param, seq))

    print(f"Total combinaciones: {len(combinations)}")
    print(f"Corriendo en paralelo con {N_JOBS} núcleos...")

    # Correr en paralelo
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_region_param)(
            region, param, data, feature_cols,
            seq, RANDOM_SEED, N_BOOTSTRAP
        )
        for region, param, seq in combinations
    )

    # Consolidar
    results = [r for r in results if r is not None]
    df_results = pd.DataFrame(results)

    # ── CORRECCIONES MÚLTIPLES ────────────────────────────────
    p_values = df_results['p_value_bootstrap'].values

    # FDR Benjamini-Hochberg
    rejected_fdr, pvals_fdr, _, _ = multipletests(
        p_values, alpha=0.05, method='fdr_bh')

    # Bonferroni
    rejected_bonf, pvals_bonf, _, _ = multipletests(
        p_values, alpha=0.05, method='bonferroni')

    df_results['p_value_fdr']        = pvals_fdr
    df_results['significant_fdr']    = rejected_fdr
    df_results['p_value_bonferroni'] = pvals_bonf
    df_results['significant_bonf']   = rejected_bonf

    # Ordenar por Balanced Accuracy
    df_results = df_results.sort_values(
        by='BalancedAcc', ascending=False)

    # Guardar
    df_results.to_csv(
        f'{output_dir}/results_exp1_{name}.csv', index=False)

    elapsed = time.time() - start
    n_sig_fdr  = rejected_fdr.sum()
    n_sig_bonf = rejected_bonf.sum()

    print(f"\n✅ {name} sujetos completado en {elapsed:.1f} segundos")
    print(f"Regiones significativas FDR:        {n_sig_fdr}/{len(combinations)}")
    print(f"Regiones significativas Bonferroni: {n_sig_bonf}/{len(combinations)}")
    print(f"\n=== TOP 5 REGIONES ({name} sujetos) ===")
    print(df_results[['Region', 'Parameter',
                       'BalancedAcc',
                       'BalancedAcc_CI_low', 'BalancedAcc_CI_high',
                       'ROC_AUC', 'MCC',
                       'p_value_bootstrap',
                       'p_value_fdr',
                       'significant_fdr',
                       'significant_bonf']].head())

# ── 7. CORRER LOS TRES CONJUNTOS ──────────────────────────────
if __name__ == '__main__':
    total_start = time.time()

    for config in configs:
        run_experiment(config, REGIONES_COMUNES)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"Tiempo total: {total_elapsed:.1f} segundos")
    print(f"{'='*60}")