addpath("/Users/jay/Documents/Jay's bits/Uni/Thesis/svmplus_matlab-master");

%% loop through 'r' train/test splits to achieve 'r' test accuracies
for r = 1:20

% load data
%path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/svmplus_matlab-master/MER/Valence/";
path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/svmplus_matlab-master/MER/Arousal/";
folder = '80_20';
load(strcat(path,folder,'/','workspace_vars/',folder,'_',num2str(r),'.mat'));

% Load whichever PI currently being tested

%percent_agree = importfile("/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Arousal/percent_agree.csv");
%percent_agree = percent_agree{:,3};
%train_pa = percent_agree(~idx);
%train_pa = train_pa';

%train_mean_col_var = train_mean_col_var';

%con_lvl_scaled = con_lvl_scaled';

rand_PI = rand_PI';

% create array of C and gamma
params = [1e-5 1e-3 1e-1 1e1 1e3 1e5 1e7 1e9;
    1e-9 1e-7 1e-5 1e-3 1e-1 1e1 1e3 1e5];


% set kernel type
kparam = struct();
kparam.kernel_type = 'gaussian';

% SVM+ 10-fold cross val coarse grid search
accs = SVM_plus_cv(params, kparam, train_features_scaled, train_labels, ...
    rand_PI);

% save coarse search results
save(strcat(path,folder,'/model_params/coarse_grid_',num2str(r)), 'accs')

% best param range informs fine grid search
param_idxs = find(accs(:,3)==max(accs(:,3)));

best_params = accs(param_idxs, :);

sz = size(best_params);

if sz(1)==1
    C=best_params(1,1);
    gam=best_params(1,2);
else
    C=mean(best_params(:,1));
    gam=mean(best_params(:,2));
end

% create array of linearly spaced param values for fine search
C_range = linspace((C*1e-2), (C*1e2), 10);
gam_range = linspace((gam*1e-2), (gam*1e2), 10);
params = [C_range; gam_range];

% repeat cross val for fine search parameters
accs = SVM_plus_cv(params, kparam, train_features_scaled, train_labels, ...
    rand_PI);

% save fine search results
save(strcat(path,folder,'/model_params/fine_grid_',num2str(r)), 'accs')

% tuned params
param_idxs = find(accs(:,3)==max(accs(:,3)));

best_params = accs(param_idxs, :);

sz = size(best_params);

if sz(1)==1
    C=best_params(1,1);
    gam=best_params(1,2);
else
    C=mean(best_params(:,1));
    gam=mean(best_params(:,2));
end

% test accuracy
acc = SVM_plus_SMO(train_features_scaled, train_labels, ...
    test_features_scaled, test_labels, 'gaussian', rand_PI, C, gam);

save(strcat(path,folder,'/model_accuracies/SVM_plus_',num2str(r)), 'acc')

clear; clc;
end    