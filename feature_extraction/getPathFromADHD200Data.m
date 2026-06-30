function y = getPathFromADHD200Data(sub_id, site_id, init_path, image_name)
% Builds the path to a FreeSurfer recon-all output file for an ADHD-200
% subject, following the layout:
%   <init_path>/<site_id>/<site_id>_<sub_id>/mri/orig/<image_name>
    y = [init_path, '/', site_id, '/', strcat(site_id, '_', sub_id), ...
        '/', 'mri', '/', 'orig', '/', image_name];
end
