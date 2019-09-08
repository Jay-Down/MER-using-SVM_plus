
%%

addpath("/Users/jay/Documents/Jay's bits/Uni/Thesis/libsvm-3.23/matlab");
%%
% calculate baseline SVM accuracies for 'r' saved train/test splits
for r = 1:100

% load data
path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/svmplus_matlab-master/MER/Valence/";
folder = '5_95';
load(strcat(path,folder,'/','workspace_vars/',folder,'_',num2str(r),'.mat'));
train_features_scaled = train_features_scaled';
test_features_scaled = test_features_scaled';

%% Cross validation for parameter determination

% create CV partitions
cv = cvpartition(length(train_features_scaled(:,1)), 'KFold', 10);

% create array of C and gamma

params = [1e-5 1e-3 1e-1 1e1 1e3 1e5 1e7 1e9;
    1e-9 1e-7 1e-5 1e-3 1e-1 1e1 1e3 1e5];

% SVM 10 fold cross val coarse grid search
% LIBSVM classifier
accs = libSVM_cv(train_features_scaled, train_labels, params);

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

% generate linearly spaced array of fine grid search ranges
C_range = linspace((C*1e-2), (C*1e2), 10);
gam_range = linspace((gam*1e-2), (gam*1e2), 10);
params = [C_range; gam_range];

% repeat cross val for fine search parameters
accs = libSVM_cv(train_features_scaled, train_labels, params);

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
opts = ['-s 0 -t 2 -c ', num2str(C), ' -g ', num2str(gam)];
            
model = svmtrain(train_labels, train_features_scaled, opts);

preds = svmpredict(test_labels, test_features_scaled, model);

acc = sum(preds == test_labels)/length(test_labels);

save(strcat(path,folder,'/model_accuracies/SVM_baseline_',num2str(r)), 'acc')

clear; clc;


end
