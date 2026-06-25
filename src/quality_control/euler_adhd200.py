import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

from config import FREESURFER_HOME, FS_INCLUDED, FS_EXCLUDED, EULER_ADHD200
os.environ['FREESURFER_HOME'] = FREESURFER_HOME
os.environ['PATH'] = f"{FREESURFER_HOME}/bin:{os.environ['PATH']}"

PATH_INCLUDED = FS_INCLUDED
PATH_EXCLUDED = FS_EXCLUDED

N_JOBS = -1  # todos los núcleos

# ── FUNCIÓN POR SUJETO ────────────────────────────────────────
def process_subject(subject, site, surf_path, group):

    results = {'subject_id': subject, 'site': site, 'group': group,
                'euler_lh': None, 'euler_rh': None}

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
                    results[f'euler_{hemi}'] = int(val)
                    break
        except Exception:
            pass

    return results

# ── RECOPILAR LISTA DE SUJETOS ────────────────────────────────
def collect_subjects(base_path, group):
    subjects = []
    for site in os.listdir(base_path):
        site_path = os.path.join(base_path, site)
        if not os.path.isdir(site_path):
            continue
        for subject in os.listdir(site_path):
            surf_path = os.path.join(site_path, subject, 'surf')
            if os.path.isdir(surf_path):
                subjects.append((subject, site, surf_path, group))
    return subjects

# ── RECOPILAR TODOS LOS SUJETOS ───────────────────────────────
print("Recopilando sujetos...")
subjects_included = collect_subjects(PATH_INCLUDED, 'Included')
subjects_excluded = collect_subjects(PATH_EXCLUDED, 'Excluded_Visual')
all_subjects = subjects_included + subjects_excluded

print(f"Incluidos:        {len(subjects_included)}")
print(f"Excluidos visual: {len(subjects_excluded)}")
print(f"Total:            {len(all_subjects)}")
print(f"Corriendo con {N_JOBS} núcleos...")

# ── CORRER EN PARALELO ────────────────────────────────────────
records = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(process_subject)(subject, site, surf_path, group)
    for subject, site, surf_path, group in all_subjects
)

df = pd.DataFrame(records)

# Euler promedio
df['euler_mean'] = (df['euler_lh'] + df['euler_rh']) / 2

print(f"\nTotal procesados: {len(df)}")
print(f"Con datos válidos: {df['euler_mean'].notna().sum()}")

# ── ANÁLISIS ESTADÍSTICO ──────────────────────────────────────
print("\n=== INCLUIDOS vs EXCLUIDOS ===")
inc = df[df['group'] == 'Included']['euler_mean'].dropna()
exc = df[df['group'] == 'Excluded_Visual']['euler_mean'].dropna()
stat, p = mannwhitneyu(inc, exc, alternative='greater')
print(f"Incluidos:  mean={inc.mean():.2f}, median={inc.median():.2f}, std={inc.std():.2f}")
print(f"Excluidos:  mean={exc.mean():.2f}, median={exc.median():.2f}, std={exc.std():.2f}")
print(f"Mann-Whitney U={stat:.1f}, p={p:.4f}")

# ── GUARDAR ───────────────────────────────────────────────────
df.to_csv(EULER_ADHD200, index=False)
print(f"\nSaved to: {EULER_ADHD200}")


