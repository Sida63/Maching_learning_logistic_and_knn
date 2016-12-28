import plot_digits
import utils
import numpy
(train_input,train_targets)=utils.load_train_small()
(valid_inputs,valid_targets)=utils.load_valid()
#plot_digits.plot_digits(train_input)
#(valid_a, valid_b)  = valid_inputs.shape
print numpy.array(train_input).shape
