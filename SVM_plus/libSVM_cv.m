function accs = libSVM_cv(train_features, train_labels, params) 

cv = cvpartition(length(train_features(:,1)), 'KFold', 10);

accs = [];

for i = 1:length(params(1, :))
    
    C = params(1,i);

    
    for j = 1:length(params(2,:))
        
        gam = params(2,j);
        
        tmp =[];
        
        for k = 1:cv.NumTestSets
            
            X_val = train_features(cv.test(k), :);
            X_train = train_features(cv.training(k), :);
            y_val = train_labels(cv.test(k));
            y_train = train_labels(cv.training(k));
            
            opts = ['-s 0 -t 2 -c ', num2str(C), ' -g ', num2str(gam)];
            
            % LIBSVM classifier
            model = svmtrain(y_train, X_train, opts);
            
            preds = svmpredict(y_val, X_val, model);
            
            acc = sum(preds == y_val)/length(y_val);
            
            tmp = [tmp; acc];
            
        end
        
        % assign mean of cross-val accuracies
        mean_acc = mean(tmp);
        accs = [accs; C gam mean_acc];
        
    end
end