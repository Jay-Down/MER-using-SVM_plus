function acc = SVM_plus_SMO(train_features, train_labels, ...
test_features, test_labels, kernel_type, train_PI, C, gam)

% calculate kernels
kparam = struct();
kparam.kernel_type = kernel_type;
[K, train_kparam] = getKernel(train_features, kparam);
testK = getKernel(test_features, train_features, train_kparam);
tK = getKernel(train_PI, kparam);
%%
% SVM+ (SMO)            
svmplus_param.svm_C = C; 
svmplus_param.gamma = gam;

% L2-SVM+ by Li et al.
model = solve_l2svmplus_kernel(train_labels, K, tK, ... 
svmplus_param.svm_C, svmplus_param.gamma);

alpha       = zeros(length(train_labels), 1);
alpha(model.SVs) = full(model.sv_coef);
alpha       = abs(alpha);
decs        = (testK + 1)*(alpha.*train_labels);
acc         = sum((2*(decs>0)-1) == test_labels)/length(test_labels);

fprintf(2, 'L2-SVM+, Acc = %.4f.\n', acc);