import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Cargar datos
from config import PHENO_ADHD200, EULER_ADHD200, PSNR_SSIM
phenotypic = pd.read_csv(PHENO_ADHD200)
df_euler   = pd.read_csv(EULER_ADHD200)
df_ps      = pd.read_csv(PSNR_SSIM)

# Euler mean
df_euler['euler_mean'] = (df_euler['euler_lh'] + df_euler['euler_rh']) / 2

# Merge ambos
df = df_euler.merge(df_ps[['subject_id', 'psnr', 'ssim']], on='subject_id', how='left')

# Solo incluidos
df_inc = df[df['group'] == 'Included'].copy()

# Extraer ID numérico del subject_id (ej. KKI_1018959 → 1018959)
df_inc['subject_id_num'] = df_inc['subject_id'].str.extract(r'(\d+)$').astype(int)

print("Ejemplo IDs extraídos:")
print(df_inc[['subject_id', 'subject_id_num']].head(5))
print()
print("Ejemplo phenotypic ScanDir ID:")
print(phenotypic['ScanDir ID'].head(5))

# Cruzar con phenotypic
df_merged = df_inc.merge(
    phenotypic[['ScanDir ID', 'DX']],
    left_on='subject_id_num',
    right_on='ScanDir ID',
    how='left'
)

print(f"\nDX valores únicos: {df_merged['DX'].unique()}")
print(f"Sin match: {df_merged['DX'].isna().sum()}")

# Binarizar DX
df_merged['dx_binary'] = df_merged['DX'].apply(
    lambda x: 'TDC' if str(x) == '0' else 'ADHD')

print(f"\nTDC:  N={len(df_merged[df_merged['dx_binary']=='TDC'])}")
print(f"ADHD: N={len(df_merged[df_merged['dx_binary']=='ADHD'])}")
print()

for metric in ['euler_mean', 'psnr', 'ssim']:
    tdc  = df_merged[df_merged['dx_binary']=='TDC'][metric].dropna()
    adhd = df_merged[df_merged['dx_binary']=='ADHD'][metric].dropna()
    stat, p = mannwhitneyu(tdc, adhd, alternative='two-sided')
    print(f'{metric}:')
    print(f'  TDC:  mean={tdc.mean():.3f} ± {tdc.std():.3f}, median={tdc.median():.3f}, N={len(tdc)}')
    print(f'  ADHD: mean={adhd.mean():.3f} ± {adhd.std():.3f}, median={adhd.median():.3f}, N={len(adhd)}')
    print(f'  U={stat:.1f}, p={p:.4f}')
    print()