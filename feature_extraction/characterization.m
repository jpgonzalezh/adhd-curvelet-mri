% =========================================================================
% characterization.m
%
% Curvelet-based feature extraction for the ADHD Curvelet MRI study.
%
% For each subject and each brain region, this script:
%   1. Loads the FreeSurfer-preprocessed brain volume and the parcellation
%      (aparc+aseg) produced by `recon-all`.
%   2. Builds a 2D mosaic (collage) of the axial slices of each region.
%   3. Applies the Fast Discrete Curvelet Transform (4 scales, 16 angles)
%      via CurveLab's fdct_wrapping, yielding 81 subbands per region.
%   4. Summarizes each subband with a Generalized Gaussian Distribution
%      (GGD), producing the 243-dimensional feature vector per region
%      (81 subbands x 3 GGD parameters) used downstream in the Python
%      analysis pipeline.
%
% Output: a CSV (one row per subject x region) whose `curv` column holds
% the 243 interleaved GGD parameters (alpha, beta, mu per subband). This
% CSV is the input expected by the Python experiments
% (curvelet_data_adhd_raw_4scales.csv).
%
% -------------------------------------------------------------------------
% REQUIREMENTS (external, not included in this repository):
%   - MATLAB (tested with R2021+), Parallel Computing Toolbox (for parfor).
%   - CurveLab 2.1.3 (fdct_wrapping_matlab):   http://www.curvelet.org
%   - Tools for NIfTI/ANALYZE (load_untouch_nii):
%       https://www.mathworks.com/matlabcentral/fileexchange/8797
%   - Local helper functions (in this folder, alongside this script):
%       getPathFromADHD200Data.m, getCollageImage.m,
%       getCurveletFeatureVector.m
%
% INPUT DATA (not distributed; see data/README.md):
%   - FreeSurfer recon-all outputs per subject (brainmask + aparc+aseg).
%   - A subject table CSV (SUBJECT_TABLE below) with columns:
%       SUB_ID, DATASET, CENTRO_NOMBRE (site), EDAD (age), DX_GROUP.
%   - region_names_ADHD.csv: region_index, region_name (FreeSurfer LUT).
%
% USAGE:
%   Edit the CONFIGURATION block below to point at your local paths,
%   then run this script. No paths are hard-coded elsewhere.
% =========================================================================

clear all;

% ----------------------------- CONFIGURATION -----------------------------
% Edit these paths to match your environment. Relative paths are resolved
% against the current MATLAB working directory.

% Folders containing the required toolboxes / helper functions:
NIFTI_TOOLBOX_PATH   = 'LibreriaNifti';            % Tools for NIfTI/ANALYZE
WAVELET_TEXTURE_PATH = 'ToolboxWaveletTexture';    % GGD fitting utilities
CURVELAB_PATH        = 'fdct_wrapping_matlab';     % CurveLab fdct_wrapping
COLLAGE_UTILS_PATH   = 'collage_utils';            % Mosaic builder helpers

% Subject table (without the .csv extension) and region lookup table:
SUBJECT_TABLE = 'DATA_ADHD_201';        % e.g. DATA_ADHD_201 / _649 / _875
REGION_TABLE  = 'region_names_ADHD.csv';

% Root folder holding the FreeSurfer-preprocessed subjects:
PREPROCESSED_ROOT = fullfile('.', 'preprocessed');   % <-- set to your path

% Folder where the output CSVs will be written:
OUTPUT_PATH = fullfile('.', 'results');              % <-- set to your path

% Volume and parcellation file names (FreeSurfer recon-all outputs):
VOLUME_NAME   = 'brainmask.nii.gz';
CORTICAL_VOL  = 'aparc+aseg.nii.gz';

% Curvelet / mosaic parameters (match the manuscript: 4 scales, 16 angles):
AXIS          = 'axial';   % axial, coronal, sagital, axial-transposed
N_SCALES      = 4;
N_ANGLES      = 16;
PADDING       = 0;
N_WORKERS     = 6;         % parallel workers for parfor

% Per-subband coefficient normalization (false in the published analysis):
COEF_NORMALIZATION = false;
% -------------------------------------------------------------------------

addpath(NIFTI_TOOLBOX_PATH);
addpath(WAVELET_TEXTURE_PATH);
addpath(CURVELAB_PATH);
addpath(COLLAGE_UTILS_PATH);

table_data   = readtable([SUBJECT_TABLE, '.csv']);
regions_data = readtable(REGION_TABLE);

table_output_name = ['curvelet_', lower(SUBJECT_TABLE), '_', AXIS];

n_regions      = height(regions_data);
num_elems_table = 8;   % columns in the output table (see table_names below)
num_subjects    = height(table_data);

if ~exist(OUTPUT_PATH, 'dir')
    mkdir(OUTPUT_PATH);
end

features_cell = cell(n_regions, num_elems_table, num_subjects);
bad_log_cell  = {};

parpool(N_WORKERS);

parfor i = 1:num_subjects
    table_row = table_data(i, :);
    sub_id    = num2str(table_row.SUB_ID);
    dataset   = cell2mat(table_row.DATASET);
    site_id   = cell2mat(table_row.CENTRO_NOMBRE);
    age       = table_row.EDAD;
    dx_group  = table_row.DX_GROUP;

    try
        volume_struc = load_untouch_nii(getPathFromADHD200Data(sub_id, ...
            site_id, PREPROCESSED_ROOT, VOLUME_NAME));

        cortical_seg_struc = load_untouch_nii(getPathFromADHD200Data(sub_id, ...
            site_id, PREPROCESSED_ROOT, CORTICAL_VOL));
    catch
        bad_sub_cell = {'file not found', sub_id, site_id, dataset};
        bad_log_cell = [bad_log_cell; bad_sub_cell];
        continue;
    end

    brain_vol = volume_struc.img;

    temporal_features_cell = cell(n_regions, num_elems_table);
    segmentation_vol = cortical_seg_struc.img;

    for j = 1:n_regions
        region_analysed = regions_data.region_index(j);

        try
            collageImage = getCollageImage(brain_vol, segmentation_vol, ...
                region_analysed, AXIS, PADDING);
            curvelets = fdct_wrapping(collageImage, 1, 1, N_SCALES, N_ANGLES);
            if COEF_NORMALIZATION
                curvelets = normalizeCurveletSubbands(curvelets);
            end
            curvelet_vector = real(getCurveletFeatureVector(curvelets));
        catch
            bad_sub_cell = {['characterization issue on region ', ...
                num2str(region_analysed)], sub_id, site_id, dataset};
            bad_log_cell = [bad_log_cell; bad_sub_cell];
            continue;
        end

        region_data_row = regions_data(j, :);
        region_name  = cell2mat(region_data_row.region_name);
        region_index = region_data_row.region_index;

        cell_row = {region_index, region_name, ...
            sub_id, dx_group, dataset, site_id, age, curvelet_vector};

        temporal_features_cell(j, :) = cell_row;
    end

    features_cell(:, :, i) = temporal_features_cell;
end

% Save the raw MATLAB cell array (intermediate, optional):
save(fullfile(OUTPUT_PATH, table_output_name), 'features_cell');

% Flatten to a table and write the CSV consumed by the Python pipeline:
table_names = {'region_index', 'region_name', 'subject_id', 'dx_group', ...
    'dataset', 'site_id', 'age', 'curv'};
out_table = cell2table(reshape(permute(features_cell, [1 3 2]), ...
    [num_subjects * n_regions, num_elems_table]));
out_table.Properties.VariableNames = table_names;

writetable(out_table, ...
    fullfile(OUTPUT_PATH, [table_output_name, '.csv']));
writetable(cell2table(bad_log_cell), ...
    fullfile(OUTPUT_PATH, [table_output_name, '-bad-log.csv']));

delete(gcp('nocreate'));
