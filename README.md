# MER Using SVM+

Music Emotion Recognition on the PMEmo dataset (http://pmemo.hellohui.cn/) employing the Learning Using Privileged Information (LUPI) paradigm via SVM+ as developed by Vapnik et al. [https://www.sciencedirect.com/science/article/pii/S0893608009001130]


Repository contains 2 folders:


## annotation_py_files

.ipynb files for segmenting music clip annotations and features.

Contents:
- **annotation_loading_segmentation.ipynb**
	annotation_loading_segmentation reads in individual annotation csv files and segments them based on affective change point locations
- **features_segmentation.ipynb**
	features_segmentation segments dynamic audio features from PMEmo dataset according to breakpoints determined in annotation_loading_segmentation


## SVM_plus

MATLAB .m files for SVM and SVM+ modelling.
  
⁃ **arousal_preproc.m**
⁃ **valence_preproc.m**
	Load segmented features and ground truths then create train/test splits, scale features, create privileged information measures, test labels created, then save workspace variables.
⁃ **libSVM_cv.m**
	Creates 10-fold cross-validated accuracies for training data - requires LIBSVM classifier [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html)
⁃ **SVM_param_tuning.m**
	Carries out coarse to fine grid search using libSVM_cv.m for cross-validation; records test accuracy
⁃ **SVM_plus_SMO.m**
	Carries out L2-SVM+ classification for train, privileged info and test data arguments
⁃ **SVM_plus_cv.m**
	Creates 10-fold cross-validated accuracies for training data using L2-SVM+
⁃ **SVM_plus_param_tuning.m**
	carries out coarse to fine grid search using libSVM_cv.m for cross 
validation; records test accuracy
⁃ **solve_l2svmplus_kernel.m**
⁃ **utils**
	contains helper functions for carrying out L2-SVM+
⁃ getKernel.m
⁃ L1_normalization.m
⁃ L2_distance_2.m
⁃ return_GaussianKernel.m


*All SVM_plus files require Li et al.’s L2-SVM+ implementation contained in solve_l2svmplus_kernel.m [SVMPLUS](https://github.com/okbalefthanded/svmplus_matlab)*



