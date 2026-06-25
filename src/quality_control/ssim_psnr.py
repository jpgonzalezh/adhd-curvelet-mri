import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import os
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

# ── CONFIGURACIÓN ─────────────────────────────────────────────
from config import FS_INCLUDED, FS_EXCLUDED, MNI_TEMPLATE, PSNR_SSIM
PATH_INCLUDED = FS_INCLUDED
PATH_EXCLUDED = FS_EXCLUDED
MNI_PATH      = MNI_TEMPLATE
N_JOBS        = -1

# ── CARGAR MNI152 UNA SOLA VEZ ────────────────────────────────
print("Cargando MNI152...")
mni_img  = nib.load(MNI_PATH)
mni_data = mni_img.get_fdata()

# Normalizar MNI a [0, 1]
mni_norm = (mni_data - mni_data.min()) / (mni_data.max() - mni_data.min())
print(f"MNI152 shape: {mni_data.shape}")

# ── FUNCIÓN POR SUJETO ────────────────────────────────────────
def process_subject(subject, site, mri_path, group, mni_norm):

    result = {
        'subject_id': subject,
        'site': site,
        'group': group,
        'psnr': None,
        'ssim': None
    }

    norm_path = os.path.join(mri_path, 'norm.mgz')
    if not os.path.exists(norm_path):
        return result

    try:
        # Cargar imagen del sujeto
        img      = nib.load(norm_path)
        img_data = img.get_fdata()

        # Verificar que tienen el mismo shape
        if img_data.shape != mni_norm.shape:
            # Redimensionar si es necesario
            from skimage.transform import resize
            img_data = resize(img_data, mni_norm.shape,
                            anti_aliasing=True)

        # Normalizar a [0, 1]
        img_norm = (img_data - img_data.min()) / \
                   (img_data.max() - img_data.min() + 1e-8)

        # Calcular PSNR
        result['psnr'] = psnr(mni_norm, img_norm, data_range=1.0)

        # Calcular SSIM
        result['ssim'] = ssim(mni_norm, img_norm,
                              data_range=1.0)

    except Exception as e:
        print(f"  Error en {subject}: {e}")

    return result

# ── RECOPILAR SUJETOS ─────────────────────────────────────────
def collect_subjects(base_path, group):
    subjects = []
    for site in os.listdir(base_path):
        site_path = os.path.join(base_path, site)
        if not os.path.isdir(site_path):
            continue
        for subject in os.listdir(site_path):
            mri_path = os.path.join(site_path, subject, 'mri')
            if os.path.isdir(mri_path):
                subjects.append((subject, site, mri_path, group))
    return subjects

print("Recopilando sujetos...")
subjects_included = collect_subjects(PATH_INCLUDED, 'Included')
subjects_excluded = collect_subjects(PATH_EXCLUDED, 'Excluded_Visual')
all_subjects      = subjects_included + subjects_excluded

print(f"Incluidos:        {len(subjects_included)}")
print(f"Excluidos visual: {len(subjects_excluded)}")
print(f"Total:            {len(all_subjects)}")
print(f"Corriendo con {N_JOBS} núcleos...")

# ── CORRER EN PARALELO ────────────────────────────────────────
records = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(process_subject)(subject, site, mri_path, group, mni_norm)
    for subject, site, mri_path, group in all_subjects
)

df = pd.DataFrame(records)

print(f"\nTotal procesados: {len(df)}")
print(f"Con PSNR válido: {df['psnr'].notna().sum()}")
print(f"Con SSIM válido: {df['ssim'].notna().sum()}")

# ── ANÁLISIS ESTADÍSTICO ──────────────────────────────────────
print("\n=== INCLUIDOS vs EXCLUIDOS ===")
for metric in ['psnr', 'ssim']:
    inc = df[df['group'] == 'Included'][metric].dropna()
    exc = df[df['group'] == 'Excluded_Visual'][metric].dropna()
    stat, p = mannwhitneyu(inc, exc, alternative='greater')
    print(f"\n{metric.upper()}:")
    print(f"  Incluidos:  mean={inc.mean():.3f} ± {inc.std():.3f}")
    print(f"  Excluidos:  mean={exc.mean():.3f} ± {exc.std():.3f}")
    print(f"  Mann-Whitney U={stat:.1f}, p={p:.4f}")

# ── GUARDAR ───────────────────────────────────────────────────
df.to_csv(PSNR_SSIM, index=False)
print(f"\nSaved to: {PSNR_SSIM}")