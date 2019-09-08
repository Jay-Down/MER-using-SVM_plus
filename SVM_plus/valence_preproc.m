% Valence
clear;
clc;

folder = '80_20';
path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/svmplus_matlab-master/MER/";

% seed random number generation for reproducible splits
rng(100);

% load features
features = readtable("/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Features/Valence_averaged_features.csv");
% remove first 5 columns of IDs and DataFrame artefacts
features = features{:,5:end};
%% load ground truths
load("/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Valence/mean_col_annotations.mat");
annotations = annotations{:,:};

thresholds = zeros(100,2);

% loop for specified number of iterations to generate i train/test splits
for i = 1:100
    % seed random number generation for reproducible splits
    rng(i)

%% Train/test feature split
    
    % generate train/test indices    
    cv = cvpartition(size(features,1),'HoldOut',0.2);
    idx = cv.test;

    train_features = features(~idx,:)';
    test_features = features(idx,:)';

    %% Feature scaling

    mins = min(train_features, [], 2);
    ranges = max(train_features, [], 2) - mins;
    
    % min-max scale feature dimensions to range [0,1]
    train_features_scaled = (train_features - repmat(mins, 1, ...
        size(train_features, 2))) ./ repmat(ranges, 1, size(train_features, 2));

    %scale test features with same scaling as train
    test_features_scaled = (test_features - repmat(mins, 1, ...
        size(test_features, 2))) ./ repmat(ranges, 1, size(test_features, 2));


    %% Target variable thresholding
    
    % ground truth train/test sets
    train_labels = annotations(~idx);
    test_labels = annotations(idx);
    
    % training annotations threshold
    thresh = mean(train_labels);
    thresholds(i,1)=i;
    thresholds(i,2)=thresh;
    
    % label generation
    train_labels(train_labels>thresh) = 1;
    train_labels(train_labels~=1) = -1;
    test_labels(test_labels>thresh) = 1;
    test_labels(test_labels~=1) = -1;

    %% Privileged Info

    PI_features = readtable("/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Valence/derived_stats.csv");
    PI_features = PI_features{:,3:end};
    PI_features(:,1)=[]; % remove redundant info

    train_PI = PI_features(~idx, :);
    test_PI = PI_features(idx, :);
    
    % consensus level
    y_votes = readtable("/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Valence/V_consensus.csv");
    y_votes = y_votes{:,2};
    train_y_votes = y_votes(~idx);
    con_lvl = zeros(length(train_y_votes), 1);

    % abs taken for fringe cases where majority consensus opposes label - count
    % no instances for report. abs has effect of ignoring class label in this
    % case whilst capturing consensus info and maintaining a split vote as no
    % consensus rather than artificially inflating 0 score if standardising
    % negative numbers
    for j = 1:length(con_lvl)
        if train_labels(j)==1
            con_lvl(j) = abs((2*0.1*train_y_votes(j))-1);
        else 
            con_lvl(j) = abs(((2*(1-(0.1*train_y_votes(j)))))-1);
        end
    end    

    mins = min(con_lvl);
    range = max(con_lvl) - mins;

    con_lvl_scaled = (con_lvl - mins)/range;
    
    % Randomly generated PI
    rand_PI = rand(size(train_labels));

    %% PI scaling

    mins = min(train_PI, [], 1);
    ranges = max(train_PI, [], 1) - mins;
    
    % scale training set of descriptive stats PI
    train_PI_scaled = (train_PI - repmat(mins, size(train_PI, 1), 1))...
        ./ repmat(ranges, size(train_PI, 1), 1);

    % mean_col_vars at col index 2
    train_mean_col_var = train_PI_scaled(:,2);
    test_mean_col_var = test_PI_scaled(:,2);

    % transpose for upcoming kernel calculations
    train_PI_scaled = train_PI_scaled';
    test_PI_scaled = test_PI_scaled';
    scaled_PI = scaled_PI';


    %% Save

    save(strcat(path,'Valence/',folder,'/','workspace_vars/',folder,'_',num2str(i)))

end
