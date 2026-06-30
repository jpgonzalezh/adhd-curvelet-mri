# Feature extraction (MATLAB)

This folder contains the MATLAB pipeline that turns FreeSurfer-preprocessed
structural MRI into the **243-dimensional Curvelet/GGD feature vector per
brain region** used by the Python analysis code in `src/`.

## What it does

For every subject and every brain region:

1. Load the FreeSurfer `recon-all` outputs (`brainmask.nii.gz` and the
   `aparc+aseg.nii.gz` parcellation).
2. Build a 2D **mosaic** of the region's axial slices.
3. Apply the **Fast Discrete Curvelet Transform** (4 scales, 16 angles)
   via CurveLab's `fdct_wrapping` → 81 subbands per region.
4. Fit a **Generalized Gaussian Distribution (GGD)** to each subband,
   producing 3 parameters (alpha, beta, mu) per subband → **243 features**
   per region (81 x 3), stored interleaved in the `curv` column.

The resulting CSV is the input expected by the Python experiments
(e.g. `curvelet_data_adhd_raw_4scales.csv`); see `data/README.md`.

## Preprocessing

All MRIs were processed beforehand with FreeSurfer **`recon-all`**
(version 7.2.0). No additional registration was applied; only the
rigid-body transforms internal to `recon-all` are used, so brain-space
metrics are preserved. The parcellation used is the standard
`aparc+aseg`, which provides cortical and subcortical labels. Of the 109
regions it produces, three (left/right vessel and optic chiasm) are excluded
because the Curvelet characterization fails in the majority of subjects,
yielding the **106 analyzed regions** reported in the paper (68 cortical +
38 subcortical and other non-cortical structures, including the left/right
cerebellar cortex, FreeSurfer labels 8 and 47).

## Requirements

### Third-party toolboxes (NOT redistributed here — download separately)

These are external packages with their own licenses. Download them and add
them to the MATLAB path (the `addpath` calls at the top of the script):

- MATLAB R2021+ with the **Parallel Computing Toolbox** (`parfor`).
- **CurveLab 2.1.3** (`fdct_wrapping_matlab`) — http://www.curvelet.org
- **Tools for NIfTI/ANALYZE** (`load_untouch_nii`) —
  https://www.mathworks.com/matlabcentral/fileexchange/8797
- **Wavelet-based texture toolbox** (GGD fitting: `ggmle`, `estpdf`,
  `sbpdf`, ...) by M. N. Do and M. Vetterli — http://www.ifp.uiuc.edu/~minhdo
  Reference: Do & Vetterli, *Wavelet-based texture retrieval using
  generalized Gaussian density and Kullback-Leibler distance*, IEEE Trans.
  Image Processing, 2002. The per-subband GGD parameters (alpha, beta, mu)
  are estimated with this package.

### Helper functions (alongside `characterization.m`)

Included in this repository, in the same folder as the main script:

- `getPathFromADHD200Data.m` — resolves a subject's recon-all file path.
- `getCollageImage.m` — dispatches to the axial mosaic builder.
- `getCollageImageAxial_square.m` — builds the square 2D mosaic for a region.
- `getCurveletFeatureVector.m` — fits a GGD (`ggmle`) to each subband and
  flattens the per-subband (alpha, beta, mu) parameters into the
  243-dimensional vector.

Required by the mosaic builder (included in the `collage_utils/` toolbox
folder and added to the path via `addpath(COLLAGE_UTILS_PATH)`):

- `regionCrop.m` — crops a region from a slice using the parcellation LUT.
- `insertMatrix.m` — inserts a matrix into a larger zero-padded canvas.

External dependency for the GGD fit:

- `ggmle` — generalized Gaussian maximum-likelihood estimator, provided by
  the **ToolboxWaveletTexture** dependency listed above.

## Inputs (not distributed — see `data/README.md`)

- FreeSurfer `recon-all` outputs per subject.
- A subject table CSV with columns
  `SUB_ID, DATASET, CENTRO_NOMBRE, EDAD, DX_GROUP`.
- `region_names_ADHD.csv` with `region_index, region_name`
  (FreeSurfer LUT codes; the file lists all 109 regions produced by
  `aparc+aseg`, of which 106 are analyzed — see note above).

## Usage

Open `characterization.m`, edit the **CONFIGURATION** block at the top to
point at your local toolbox and data paths, then run the script. All paths
live in that block; nothing is hard-coded elsewhere. Outputs (the feature
CSV and a `-bad-log.csv` listing any subjects/regions that failed) are
written to `OUTPUT_PATH`.
