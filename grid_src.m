load('model_6fold.mat')

% subjects
cfg.n_subj = 27;

% Define some parameters to keep track of progress
cfg.n_jobs = cfg.n_subj; % same of subj number
counter_job = 0;
warning('off','all')
start_time = datestr(datetime('now'));
start_time = start_time([13:20]);

for subj = cfg.n_subj
    
    subj_fn = ['4_glm\subject_',num2str(subj),'/SPM.mat'];
    ds = cosmo_fmri_dataset(subj_fn);
    
    ds.sa.targets = repmat(1:17,1,8)';
    ds.sa.chunks = floor((((1:136)-1)/17)+1)';
    idx_directions = find (ds.sa.targets <= 16);
    ds = cosmo_slice(ds,idx_directions,1);
    
    % compute average for each unique target
    ds=cosmo_fx(ds, @(x)mean(x,1), 'targets', 1);
    ds.samples = bsxfun(@minus, ds.samples, mean(ds.samples, 1));
    ds = cosmo_remove_useless_data(ds);

    % construct models
    my_model = cosmo_squareform(model_6fold);
    
    % Set general parameters for correlation measure
    measure = @cosmo_target_dsm_corr_measure;
    measure_args = struct();
    measure_args.target_dsm = my_model;
    measure_args.center_data=true;
    
    % Set parameters for spherical rois
    radius = 3; % voxels
    
    % Define a neighborhood of rois where to perform the analysis
    nbrhood=cosmo_spherical_neighborhood(ds,'radius',radius);
    
    % Run the searchlight
    results = cosmo_searchlight(ds,nbrhood,measure,measure_args);
    
    % Save the results in a .nii file
    cosmo_map2fmri(results, ...
        ['results_grid_src/results_grid_src_subject_',num2str(subj),'.nii']);
   
end
