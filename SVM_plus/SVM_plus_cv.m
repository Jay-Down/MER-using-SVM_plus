function accs = SVM_plus_cv(params, kparam, train_features, train_labels, train_PI)

accs = [];

% create CV partitions
cv = cvpartition(length(train_features(1,:)), 'KFold', 10);

% cross-val 
for i = 1:length(params(1, :))
    
    C = params(1,i);
    
    for j = 1:length(params(2,:))
        
        gam = params(2,j);
        
        tmp =[];
        
        for k = 1:cv.NumTestSets
            
            X_train = train_features(:,cv.training(k));
            y_train = train_labels(cv.training(k));
            
            X_val = train_features(:, cv.test(k));            
            y_val = train_labels(cv.test(k));
            
            PI_fold = train_PI(:, cv.training(k));
            
            % calculate kernels
            [K, train_kparam] = getKernel(X_train, kparam);
            testK = getKernel(X_val, X_train, train_kparam);
            tK = getKernel(PI_fold, kparam);
            
            % SVM+ (SMO)
            svmplus_param.svm_C = C; 
            svmplus_param.gamma = gam;
            
            % L2_SVM+ by Li et al.
            model = solve_l2svmplus_kernel(y_train, K, tK, ... 
            svmplus_param.svm_C, svmplus_param.gamma);
        
            alpha       = zeros(length(y_train), 1);
            alpha(model.SVs) = full(model.sv_coef);
            alpha       = abs(alpha);
            decs        = (testK + 1)*(alpha.*y_train);
            acc         = sum((2*(decs>0)-1) == y_val)/length(y_val);

            fprintf(2, 'L2-SVM+, Acc = %.4f.\n', acc);
            

            tmp = [tmp; acc];
            
        end
        
        % assign mean of cross-val accuracies
        mean_acc = mean(tmp);
        accs = [accs; C gam mean_acc];
        
    end
end