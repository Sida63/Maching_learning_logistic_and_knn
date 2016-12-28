from run_knn import run_knn
import utils
import numpy as np
import matplotlib.pyplot as plt
import plot_digits
(train_input,train_targets)=utils.load_train()
(valid_inputs,valid_targets)=utils.load_valid()
(test_inputs,test_targets)=utils.load_test()
(valid_inputr, valid_inputc)  = valid_inputs.shape
(test_inputr, test_inputc)  = test_inputs.shape
k               = np.zeros(5)
classificationrate_valid   = np.zeros(5)
classificationrate_test    = np.zeros(5)
for i in range(5):
    k[i] = 2*i+1;
    valid_count=0
    test_count=0
    valid_p = run_knn(k[i],train_input,train_targets,valid_inputs)
    (row,col)=valid_p.shape
    for j in xrange(row):
        if valid_p[j][0]==valid_targets[j][0]:
            valid_count+=1
    classificationrate_valid[i]=valid_count*1.0/float(valid_inputr)
    test_p = run_knn(k[i], train_input, train_targets, test_inputs)
    (row,col)=test_p.shape
    for j in xrange(row):
        if test_p[j][0]==test_targets[j][0]:
            test_count+=1
    classificationrate_test[i]=test_count*1.0/float(test_inputr)


print classificationrate_valid
print classificationrate_test

plt.plot(k, classificationrate_valid, marker='o', label='Validation Set')
plt.plot(k, classificationrate_test, marker='x', label='Test Set')
legend = plt.legend(loc=3)
plt.xlabel('k')
plt.ylabel('Classification Rate')
plt.axis([1, 9, 0.7, 1])
plt.show()

#plot_digits.plot_digits(train_input)


