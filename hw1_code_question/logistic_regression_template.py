from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt
import copy


def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()
    train_inputs_small, train_targets_small=load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs,test_targets=load_test()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape
    Ns,Ms= train_inputs_small.shape
    # Logistic regression weights
    # TODO:Initialize to random weights here.
    #weights = 0.05*np.ones((M+1,1))
    weights = 0.1*np.random.randn(M+1,1)
    weightssmall=copy.deepcopy(weights)
    weightspen=copy.deepcopy(weights)
    weightspensmall=copy.deepcopy(weights)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 9))
    loggingsmall= np.zeros((hyperparameters['num_iterations'], 9))
    loggingpen= np.zeros((hyperparameters['num_iterations'], 9))
    loggingpensmall= np.zeros((hyperparameters['num_iterations'], 9))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        validf, validdf, validpredictions = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        predictions_test=logistic_predict(weights, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)

        # print some stats
       # print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
              # "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                #   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                #   float(cross_entropy_valid), float(frac_correct_valid*100))
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100,f,validf]
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weightssmall, train_inputs_small, train_targets_small, hyperparameters)
        validf, validdf, validpredictions = logistic(weightssmall, valid_inputs, valid_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets_small, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weightssmall = weightssmall - hyperparameters['learning_rate'] * df / Ns

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weightssmall, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        predictions_test=logistic_predict(weightssmall, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)


        # print some stats
        #print "this is for smalltrainset"
        #print ("ITERATION:{:4d}  SMALLTRAIN NLOGL:{:4.2f}  SMALLTRAIN CE:{:.6f} "
               #"SMALLTRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                  # t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                  # float(cross_entropy_valid), float(frac_correct_valid*100))
        loggingsmall[t] = [f / Ns, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100,f,validf]
    hyperparameters['weight_regularization']=True
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weightspen, train_inputs, train_targets, hyperparameters)
        validf, validdf, validpredictions = logistic(weightspen, valid_inputs, valid_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weightspen = weightspen - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weightspen, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        predictions_test=logistic_predict(weightspen, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)

        # print some stats
       # print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
              # "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                #   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                #   float(cross_entropy_valid), float(frac_correct_valid*100))
        loggingpen[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100,f,validf]
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weightspensmall, train_inputs_small, train_targets_small, hyperparameters)
        validf, validdf, validpredictions = logistic(weightspensmall, valid_inputs, valid_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets_small, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weightspensmall = weightspensmall - hyperparameters['learning_rate'] * df / Ns

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weightspensmall, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        predictions_test=logistic_predict(weightspensmall, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)


        # print some stats
        #print "this is for smalltrainset"
        #print ("ITERATION:{:4d}  SMALLTRAIN NLOGL:{:4.2f}  SMALLTRAIN CE:{:.6f} "
               #"SMALLTRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                  # t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                  # float(cross_entropy_valid), float(frac_correct_valid*100))
        loggingpensmall[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100,f,validf]
        
    return logging,loggingsmall,loggingpen,loggingpensmall

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.4,
                    "weight_regularization":False,# boolean, True for using Gaussian prior on weights
                    'num_iterations': 50,
                    'weight_decay': 0.01 # related to standard deviation of weight prior 
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 9))
    loggingsmall = np.zeros((hyperparameters['num_iterations'], 9))
    loggingpen = np.zeros((hyperparameters['num_iterations'], 9))
    loggingpensmall = np.zeros((hyperparameters['num_iterations'], 9))
    for i in xrange(num_runs):
        temp,temp1,temp2,temp3=run_logistic_regression(hyperparameters)
        logging += temp
        loggingsmall+=temp1
        loggingpen+=temp2
        loggingpensmall+=temp3
    logging /= num_runs
    loggingsmall /= num_runs
    loggingpen/=num_runs
    loggingpensmall/=num_runs
    #print logging[:,1]
    # TODO generate plots
    print logging[len(logging)-1,1],logging[len(logging)-1,3],logging[len(logging)-1,5]
    print loggingpen[len(logging)-1,1],logging[len(logging)-1,3],logging[len(logging)-1,5]
    print loggingsmall[len(loggingsmall)-1,1],loggingsmall[len(loggingsmall)-1,3],loggingsmall[len(loggingsmall)-1,5]

    

    print "rate"
    print 1-(logging[len(logging)-1,2]/100),1-(logging[len(logging)-1,4]/100),1-(logging[len(logging)-1,6]/100)
    print 1-loggingpen[len(logging)-1,2]/100,1-logging[len(logging)-1,4]/100,1-logging[len(logging)-1,6]/100
    
    plt.figure(figsize=(8,7),dpi=98)

    plt.plot(xrange(len(logging)), logging[:,1], marker='o', label='Training Set')
    plt.plot(xrange(len(logging)), logging[:,3], marker='x', label='Validation Set')
    #plt.plot(xrange(len(logging)), logging[:,5], marker='.', label='Test Set')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(logging), 0, 300])
    plt.show()


    plt.plot(xrange(len(loggingsmall)), loggingsmall[:,1], marker='o', label='SmallTraining Set')
    plt.plot(xrange(len(loggingsmall)), loggingsmall[:,3], marker='x', label='Validation Set')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 300])
    plt.show()

    plt.plot(xrange(len(logging)), logging[:,6]/100, marker='o', label='Training Set')
    plt.plot(xrange(len(loggingpen)), loggingpen[:,6]/100, marker='x', label='Trainingwithrigu Set')
    plt.ylabel('Test Correct Rate')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 1])
    plt.show()

    plt.plot(xrange(len(loggingsmall)), loggingsmall[:,6]/100, marker='o', label='SmallTraining Set')
    plt.plot(xrange(len(loggingpensmall)), loggingpensmall[:,6]/100, marker='x', label='SmallTrainingwithrigu Set')
    plt.ylabel('Test Correct Rate')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 1])
    plt.show()
 
    plt.plot(xrange(len(logging)), logging[:,1], marker='o', label='Training Set')
    plt.plot(xrange(len(loggingsmall)), loggingsmall[:,1], marker='x', label='SmallTraining Set')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 300])
    plt.show()
    
    plt.plot(xrange(len(logging)), 1-logging[:,2]/100, marker='o', label='Training Set')
    plt.plot(xrange(len(logging)), 1-logging[:,4]/100, marker='x', label='Validation Set')
    plt.plot(xrange(len(logging)), 1-logging[:,6]/100, marker='.', label='Test Set')
    plt.ylabel('Error Rate')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 1])
    plt.show()

    plt.plot(xrange(len(loggingsmall)), 1-loggingsmall[:,2]/100, marker='o', label='SmallTraining Set')
    plt.plot(xrange(len(loggingsmall)), 1-loggingsmall[:,4]/100, marker='x', label='Validation Set')
    plt.plot(xrange(len(loggingsmall)), 1-loggingsmall[:,6]/100, marker='.', label='Test Set')
    plt.ylabel('Error Rate')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 1])
    plt.show()

    plt.plot(xrange(len(loggingpen)), logging[:,7], marker='o', label='Training Set')
    plt.plot(xrange(len(loggingpen)), logging[:,8], marker='x', label='Validation Set')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 500])
    plt.show()

    plt.plot(xrange(len(loggingsmall)), loggingpensmall[:,7], marker='o', label='SmallTraining Set')
    plt.plot(xrange(len(loggingsmall)), loggingpensmall[:,8], marker='x', label='Validation Set')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.axis([1, len(loggingsmall), 0, 500])
    plt.show()

