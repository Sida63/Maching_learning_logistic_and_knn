""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    (n,m) = data.shape
    x0 = np.ones((n,1))
    #Add one colunm 1 to make data and weights have same number of column
    #axis=1 means operating column 
    newdata = np.concatenate((data,x0),axis=1)
    wx = newdata.dot(weights)
    y = sigmoid(wx)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    #reshape(-1,1) change n*1 metrix to 1*n metrix
    ce = -1 * np.sum(targets * np.log(np.array(y).reshape(-1,1)) + (1 - targets) * np.log(1 - np.array(y).reshape(-1,1)))
    #ce = -1 * np.sum(targets * np.log(np.array(y).reshape(-1,1)))
    
    #frac_correct
    #calculate the correct rate through the number of correctness(if probablity is larger than 0.5) devided by the total samples
    c = 0
    for i in xrange(len(targets)):
        if targets[i] == round(y[i]):
            c += 1
    frac_correct = float(c) / len(targets)
    

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    #judge whether we want to use Regularized logistic regression 
    if hyperparameters['weight_regularization'] is True:
        #get the probablities
        y = logistic_predict(weights, data)
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        (n,m)=data.shape
        x0 = np.ones((n,1))
        newdata = np.concatenate((data,x0),axis=1)
        y = logistic_predict(weights,data)
        f =  -np.dot(targets.T, np.log(y))-np.dot(1-targets.T, np.log(1-y))
        df = np.dot(newdata.T,y-targets)


    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    #print "i'm working"
    (n,m)=data.shape
    x0 = np.ones((n,1))
    newdata = np.concatenate((data,x0),axis=1)
    y = logistic_predict(weights,data)
    alpha = hyperparameters['weight_decay']
    
    #f is the loss function plus alpha/2 * sigma (wi^2)
    f =  -np.dot(targets.T, np.log(y))-np.dot(1-targets.T, np.log(1-y)) +0.5*alpha*(np.dot(weights.T,weights))
    #df is the df plus lambda wi accroding to ppt
    df = np.dot(newdata.T,y-targets) + alpha * weights

    
    return f, df
