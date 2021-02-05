% you need MATLAB and CoSMoMVPa (http://cosmomvpa.org)
% load the model with the periodicity you want to test (e.g., our main hp was on a 6-fold rotational symmetry typical of grid-cells)
load('model_6fold.mat')

% number of subjects
cfg.n_subj = 27;

% path to the masks
cfg.roi_path = [cd,'/6_masks/'];
cfg.roi = {'rpmEC_PHCpref_left_MNI.nii';...
    'rpmEC_PHCpref_right_MNI.nii'};

% define some parameters to keep track of progress (optional)
cfg.n_jobs = cfg.n_subj*length(cfg.roi);
job = 0;
warning('off','all')
clock_start = clock();
prev_msg ='';
start_time = datestr(datetime('now'));
start_time = start_time([13:20]);

% pre-allocate rois (this speeds up the analysis)
n_roi = length(cfg.roi);
for r = 1:n_roi
    roi_fn = [cd,'/6_masks/',cfg.roi{r},];
    ds_roi{r} = cosmo_fmri_dataset(roi_fn); 
end

% open for loop across subjects
for subj =1:cfg.n_subj
    
    % define path to the subject SPM.mat file and load it as dataset (ds) on cosmomvpa
    subj_fn = ['4_glm\subject_',num2str(subj),'/SPM.mat'];
    ds = cosmo_fmri_dataset(subj_fn);
    
    % select betas for the 16 directions (please refer to the manual on cosmomvpa to know how to treat datasets
    ds.sa.targets = repmat(1:17,1,8)';
    ds.sa.chunks = floor((((1:136)-1)/17)+1)';
    idx_directions = find (ds.sa.targets <= 16);
    ds = cosmo_slice(ds,idx_directions,1);
    
    % open for loop across rois
    for r = 1:length(cfg.roi)
        
        % mask this subjects' ds with this roi
        ds_masked = cosmo_slice(ds,logical(ds_roi{r}.samples),2);
        
        % compute average for each unique target across runs
        ds_masked=cosmo_fx(ds_masked, @(x)mean(x,1), 'targets', 1);
        ds_masked.samples = bsxfun(@minus, ds_masked.samples, mean(ds_masked.samples, 1));
        ds_masked = cosmo_remove_useless_data(ds_masked);
        
        % update your progress on the command window (optional)
        job = job+1;
        ratio_done = job/cfg.n_jobs;
        status = sprintf('done %.1f%%', ratio_done*100);
        prev_msg = cosmo_show_progress(clock_start,ratio_done,status,prev_msg);
        
        % vectorize the model
        my_model = cosmo_squareform(model_6fold);
        
        % neural matrix with Pearson's correlation
        dsm = cosmo_pdist(ds_masked.samples,'correlation');
        
        % put together neural and predicted matrices
        dsms = [dsm',my_model'];
        
        % RSA
        cc = cosmo_corr(dsms);
        
        % store results
        results(subj,r) = cc(1,2);
 
    end
end

% please cite Viganò, Rubino, Di Soccio, Buiatti, Piazza (XXX) Grid-like and distance codes for representing word meaning in the human brain, XXX, XXX:XXX.
