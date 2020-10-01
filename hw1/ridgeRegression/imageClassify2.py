
from mnist import MNIST
import numpy as np
import math
from matplotlib import pyplot as plt

class ImageClassification:


    def __init__(self, p = 100, reg_lambda = 1e-4):
        self.p = p
        self.reg_lambda = reg_lambda
        self.G = None
        self.b = None
        self.W = None

    def train(self, X, X_labels, reg_lambda=0):
        # TODO: polyfeature + normalization?
        # X = self.pFeature(X)
        n = X.shape[1]
        Y = np.eye(10)[X_labels]

        # analytical solution
        self.W = np.linalg.pinv(X.T.dot(X) + np.multiply(self.reg_lambda, np.eye(n))).dot(X.T).dot(Y)



    def predict(self, h):
        predictions = np.eye(10).T.dot(self.W.T).dot(h.T)
        return predictions



    def pFeature(self, X_train, X_test, new=True):
        train_idx = len(X_train)
        X = np.concatenate( (X_train, X_test), axis = 0)
        n, d = X.shape
        if new:
            # if self.G is None:
            self.G = np.random.normal(0, 0.3162, size=(self.p, d))  # sigma^2 = 0.1, tae a approximate value here
            # if self.b is None:
            self.b = np.random.uniform(0, math.pi, size=(n, self.p))
        h = np.cos(X.dot(self.G.T) + self.b)
        h_train = h[:train_idx]
        h_test = h[train_idx:]
        return h_train, h_test




def load_dataset():
    mndata = MNIST("./python-mnist/data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test =  map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test



def plotLearningCurve(errorTrain, errorTest, ps, regLambda):
    """
        plot computed learning curve
    """
    minX = 0
    maxY = max(errorTest)

    xs = np.arange(len(errorTrain))
    plt.plot(ps, errorTrain, 'r-o')
    plt.plot(ps, errorTest, 'b-o')
    plt.plot(ps, np.ones(len(xs)), 'k--')
    plt.legend(['Training Error', 'Validation Error'], loc='best')
    plt.title('Learning Curve (lambda='+str(regLambda)+')')
    plt.xlabel('p value')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim(top=maxY)
    # plt.xlim((minX, 10))
    plt.show()


def splitTrainValidation(X, X_labels):
    n = len(X)
    indices = np.random.permutation(n)
    training_idx_num = int(n * 0.8)
    training_idx, validation_idx = indices[:training_idx_num], indices[training_idx_num:]
    training_data = X[training_idx, :]
    training_labels = X_labels[training_idx]
    val_data = X[validation_idx, :]
    val_labels = X_labels[validation_idx]
    return training_data, training_labels, val_data, val_labels

def hoeffding( num_samples, num_correct, a= 0, b= 1, delta = 0.05):
    rhs = math.sqrt( (b-a)**2*math.log(2/delta, math.e)/(2* num_samples) )
    mu_conf_interval = ( -rhs+test_error ,  rhs+test_error )
    return mu_conf_interval


if __name__ == '__main__':


    X_train, labels_train, X_test, labels_test = load_dataset()

    ps = np.arange(500,3000, 500)  # p values
    train_errors = np.zeros(len(ps)) # training error place holder
    val_errors = np.zeros(len(ps))
    idx = 0
    reg_lambda = 1e-4

    models = []


    # for loop
    for p in ps:
        model = ImageClassification(p)
        # feature transform, each model create a new p feature
        X_train_h, X_test_p = model.pFeature(X_train, X_test)  # when transform features
        # split train and validation dataset
        X_train_h, labels_train_h, X_val_h, val_labels = splitTrainValidation(X_train_h, labels_train)


        w = model.train(X_train_h, labels_train_h, reg_lambda)


        # training error
        prediction_train = model.predict(X_train_h)
        prediction_train = np.argmax(prediction_train, axis=0)
        train_errors[idx] = 1 - np.sum(labels_train_h == np.array(prediction_train)) / len(labels_train_h)
        print("Train error:  ", 1 - np.sum(labels_train_h == np.array(prediction_train)) / len(labels_train_h))


        # validation error
        prediction_val = model.predict(X_val_h)
        prediction_val = np.argmax(prediction_val, axis=0)
        val_errors[idx] = 1 - np.sum(val_labels == np.array(prediction_val)) / len(val_labels)
        print("Validation error:   ", 1 - np.sum(val_labels == np.array(prediction_val)) / len(val_labels))



        idx += 1
        print("Training progress:  ", idx, "/", len(ps), "     current p value:", p)
        print("")
        models.append(model)

    plotLearningCurve(train_errors, val_errors, ps, reg_lambda)

    print("best model index: ", np.argmin(val_errors))
    best_model = models[np.argmin(val_errors)]


    _, h_test = best_model.pFeature(X_train, X_test, new=False)
    prediction_test = best_model.predict(h_test)
    prediction_test = np.argmax(prediction_test, axis=0)
    test_error = 1 - np.sum(labels_test == np.array(prediction_test)) / len(labels_test)
    num_samples = len(X_test)
    num_correct = np.sum(labels_test == np.array(prediction_test))
    print("Test error : ", test_error)
    print("Confidence interval:  ", hoeffding(num_samples, test_error))






