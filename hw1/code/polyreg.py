'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import math

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1e-6):
        """
        Constructor
        """
        #TODO
        self.degree = degree
        self.reg_lambda = reg_lambda
        self.theta = None
        self.featureMean = None
        self.featureStd= None

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        #TODO
        n = X.shape[0]
        zero_matrix = np.zeros((n, degree))
        zero_matrix[:, 0] = X.reshape(n)
        X = zero_matrix
        for d in range(1, degree + 1):
            X[:, d - 1] = np.power(X[:, 0], d)
        return X



    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        #TODO
        X = self.polyfeatures(X, self.degree)
        self.featureMean = X.mean(axis = 0)
        self.featureStd = np.std(X,axis = 0)

        # data standarzation
        X  = (X - self.featureMean)/ self.featureStd


        # old
        # add 1s column
        n = len(X)
        X_ = np.c_[np.ones([n, 1]), X]

        n, d = X_.shape
        d = d-1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO
        # standardize data
        X = self.polyfeatures(X, self.degree)
        X = (X - self.featureMean) / self.featureStd


        # copied code
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]
        print("reg_lambda: ", self.reg_lambda, "    model theta L2 norm: ", np.linalg.norm(self.theta, 2))
        return X_.dot(self.theta)

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)
    m = len(Xtest)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    for i in range(n):
        if i < 3:   # ignore the zero-th case
            errorTrain[i] = 0
            errorTest[i] = 0
        else:
            model = PolynomialRegression(degree, reg_lambda)
            model.fit(Xtrain[:i+1], Ytrain[:i+1])  # i = 3,  model training on sample 0,1,2
            errorTrain[i] =(1/i) * np.sum((Ytrain[:i] - model.predict(Xtrain[:i])) ** 2)
            errorTest[i] = (1/m) * np.sum((Ytest - model.predict(Xtest)) ** 2)

    return errorTrain, errorTest
