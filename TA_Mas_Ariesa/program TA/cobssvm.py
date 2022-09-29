from libsvm.svmutil import *

y, x = svm_read_problem('datasetSVR2')
m = svm_train(y, x, '-s 3 -t 2')
svm_save_model('modelSVR1', m)
