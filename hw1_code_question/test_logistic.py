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
    logging = np.zeros((hyperparameters['num_iterations'], 7))
    loggingsmall= np.zeros((hyperparameters['num_iterations'], 7))
    loggingpen= np.zeros((hyperparameters['num_iterations'], 7))
    loggingpensmall= np.zeros((hyperparameters['num_iterations'], 7))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
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
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100]
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weightssmall, train_inputs_small, train_targets_small, hyperparameters)
        
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
        loggingsmall[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100,cross_entropy_test,frac_correct_test*100]
        
    return logging,loggingsmall

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
                    "weight_regularization":True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 50,
                    'weight_decay': 1 # related to standard deviation of weight prior 
                    }

    # average over multiple runs
    num_runs = 1
    ceresult=[]
    clresult=[]
    fresult=[]
    
    for h in [0.001,0.01,0.1,1]:
        hyperparameters['weight_decay']=h
        logging = np.zeros((hyperparameters['num_iterations'], 7))
        loggingsmall = np.zeros((hyperparameters['num_iterations'], 7))
        for i in xrange(num_runs):
            temp,temp1=run_logistic_regression(hyperparameters)
            logging += temp
            loggingsmall+=temp1
        logging /= num_runs
        loggingsmall /= num_runs
        ceresult.append([logging[49,1],logging[49,3],logging[49,5]])
        clresult.append([logging[49,2],logging[49,4],logging[49,6]])
        fresult.append(logging[49,0])
    #print logging[:,1]
    # TODO generate plots
    #result.append([logging[39,1],logging[39,3],logging[39,5]])
    #print logging[39,2],logging[39,4],logging[39,6]
    print clresult
    print fresult
    plt.figure(figsize=(8,7),dpi=98)

    plt.plot([0.001,0.01,0.1,1], [ceresult[0][0],ceresult[1][0],ceresult[2][0],ceresult[3][0]], marker='o', label='Training Set')
    plt.plot([0.001,0.01,0.1,1], [ceresult[0][1],ceresult[1][1],ceresult[2][1],ceresult[3][1]], marker='x', label='Validation Set')
    plt.plot([0.001,0.01,0.1,1], [ceresult[0][2],ceresult[1][2],ceresult[2][2],ceresult[3][2]], marker='.', label='Test Set')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='upper right')
    plt.xlabel('alpha')
    plt.axis([0, 1, 10, 15])
    plt.show()

    plt.plot([0.001,0.01,0.1,1], [clresult[0][0]/100,clresult[1][0]/100,clresult[2][0]/100,clresult[3][0]/100], marker='o', label='Training Set')
    plt.plot([0.001,0.01,0.1,1], [clresult[0][1]/100,clresult[1][1]/100,clresult[2][1]/100,clresult[3][1]/100], marker='x', label='Validation Set')
    plt.plot([0.001,0.01,0.1,1], [clresult[0][2]/100,clresult[1][2]/100,clresult[2][2]/100,clresult[3][2]/100], marker='.', label='Test Set')
    plt.ylabel('Rate')
    plt.legend(loc='upper right')
    plt.xlabel('alpha')
    plt.axis([0, 1, 0.9, 1])
    plt.show()

    plt.plot([0.001,0.01,0.1,1], [fresult[0],fresult[1],fresult[2],fresult[3]], marker='o', label='Training Set')
    plt.ylabel('F/n')
    plt.legend(loc='upper right')
    plt.xlabel('alpha')
    plt.axis([0, 1, 0, 0.2])
    plt.show()
