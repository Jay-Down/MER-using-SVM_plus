Directory contains 2 folders:

	•	annotation_py_files: .py and .ipynb files for segmenting annotations and features
	⁃	annotation_loading_segmentation.ipynb/.py
	⁃	features_segmentation.ipynb/.py

annotation_loading_segmentation reads in individual annotation csv files and segments them based on affective change point locations.

features_segmentation segments dynamic audio features from PMEmo dataset according to breakpoints determined in annotation_loading_segmentation.

.py and .ipynb files provided for both.


	•	SVM_plus: MATLAB .m files for SVM and SVM+ modelling 
	⁃	arousal_preproc.m
	⁃	valence_preproc.m
	⁃	libSVM_cv.m
	⁃	SVM_param_tuning.m
	⁃	SVM_plus_SMO.m
	⁃	SVM_plus_cv.m
	⁃	SVM_plus_param_tuning.m
	⁃	solve_l2svmplus_kernel.m
	⁃	utils
	⁃	getKernel.m
	⁃	L1_normalization.m
	⁃	L2_distance_2.m
	⁃	return_GaussianKernel.m

arousal_ and valence_preproc.m load segmented features and ground truths then create train/test splits, scale features, create privileged information measures, test labels created, then save workspace variables.

libSVM_cv.m creates 10-fold cross-validated accuracies for training data - requires LIBSVM classifier (https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html).

SVM_param_tuning carries out coarse to fine grid search using libSVM_cv.m for cross-validation; records test accuracy.

SVM_plus_SMO.m function carries out L2-SVM+ classification for train, privileged info and test data arguments.

SVM_plus_cv.m creates 10-fold cross-validated accuracies for training data using L2-SVM+

SVM_plus_param_tuning.m carries out coarse to fine grid search using libSVM_cv.m for cross 
validation; records test accuracy.

*** All SVM_plus files require Li et al.’s L2-SVM+ implementation contained in solve_l2svmplus_kernel.m (https://github.com/okbalefthanded/svmplus_matlab)*** 

Utils contains helper functions for carrying out L2-SVM+.

