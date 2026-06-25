import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

from config import FREESURFER_HOME, FS_WMR, PARTICIPANTS_WMR, EULER_WMR
os.environ['FREESURFER_HOME'] = FREESURFER_HOME
os.environ['PATH'] = f"{FREESURFER_HOME}/bin:{os.environ['PATH']}"

BASE_PATH  = FS_WMR
PARTS_CSV  = PARTICIPANTS_WMR

# ── FUNCIÓN POR SUJETO ────────────────────────────────────────
def process_subject(subject_id, subject_num, dx):
    surf_path = os.path.join(BASE_PATH, f'ADHD_{subject_num}', 'surf')
    result = {
        'subject_id': subject_id,
        'subject_num': subject_num,
        'dx': dx,
        'euler_lh': None,
        'euler_rh': None
    }

    for hemi in ['lh', 'rh']:
        fpath = os.path.join(surf_path, f'{hemi}.orig.nofix')
        if not os.path.exists(fpath):
            continue
        try:
            out = subprocess.run(
                [f'{FREESURFER_HOME}/bin/mris_euler_number', fpath],
                capture_output=True, text=True, timeout=30
            )
            for line in out.stdout.split('\n') + out.stderr.split('\n'):
                if 'euler' in line.lower() and '=' in line:
                    val = line.split('=')[-1].strip().split()[0]
                    result[f'euler_{hemi}'] = int(val)
                    break
        except Exception:
            pass

    return result

# ── CARGAR PARTICIPANTES ──────────────────────────────────────
df_parts = pd.read_csv(PARTS_CSV)

# Extraer número del participant_id (sub-01 → 1)
df_parts['subject_num'] = df_parts['participant_id'].str.extract(r'(\d+)$').astype(int)

print(f"Total sujetos WMR-ADHD: {len(df_parts)}")
print(f"TDC: {len(df_parts[df_parts['ADHD_diagnosis']==0])}")
print(f"ADHD: {len(df_parts[df_parts['ADHD_diagnosis']==1])}")

# ── CORRER EN PARALELO ────────────────────────────────────────
records = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_subject)(
        row['participant_id'],
        row['subject_num'],
        row['ADHD_diagnosis']
    )
    for _, row in df_parts.iterrows()
)

df_wmr = pd.DataFrame(records)
df_wmr['euler_mean'] = (df_wmr['euler_lh'] + df_wmr['euler_rh']) / 2

print(f"\nCon datos válidos: {df_wmr['euler_mean'].notna().sum()}")

# ── ANÁLISIS TDC vs ADHD ──────────────────────────────────────
print("\n=== EULER NUMBER WMR-ADHD — TDC vs ADHD ===")
for dx, name in [(0,'TDC'), (1,'ADHD')]:
    g = df_wmr[df_wmr['dx']==dx]['euler_mean'].dropna()
    print(f"{name}: mean={g.mean():.2f}, median={g.median():.2f}, std={g.std():.2f}, N={len(g)}")

tdc  = df_wmr[df_wmr['dx']==0]['euler_mean'].dropna()
adhd = df_wmr[df_wmr['dx']==1]['euler_mean'].dropna()
stat, p = mannwhitneyu(tdc, adhd, alternative='two-sided')
print(f"\nMann-Whitney U={stat:.1f}, p={p:.4f}")

# ── GUARDAR ───────────────────────────────────────────────────
df_wmr.to_csv(EULER_WMR, index=False)
print(f"\nSaved to: {EULER_WMR}")