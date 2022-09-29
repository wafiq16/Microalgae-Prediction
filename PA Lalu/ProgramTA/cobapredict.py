from libsvm.svmutil import *
import numpy as np

m = svm_load_model('modelSVR1')
x = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
p_label, p_acc, p_val = svm_predict([], x, m)
print(p_label)

