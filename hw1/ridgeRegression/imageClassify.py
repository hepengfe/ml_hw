from mnist import MNIST
import numpy as np
def load_dataset():
    mndata = MNIST("./python-mnist/data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test
def predict(w, x):
    predictions = np.eye(10).T.dot(w.T).dot(x.T)
    return predictions




def train(X, X_labels, reg_lambda = 0):
    # TODO: polyfeature + normalization?

    n = X.shape[1]
    Y = np.eye(10)[X_labels]
    w = np.linalg.pinv(X.T.dot(X) + np.multiply(reg_lambda, np.eye(n))).dot(X.T).dot(Y)
    #     w = np.linalg.pinv(X.T.dot(X) + np.multiply(reg_lambda,np.eye(n)) ).dot(X.T).dot(Y)
    return w


if __name__ == '__main__':
    X_train, labels_train, X_test, labels_test = load_dataset()
    reg_lambda = 1e-4
    w = train(X_train, labels_train, reg_lambda)
    prediction_train = predict(w, X_train)
    prediction_train = np.argmax(prediction_train, axis=0)
    print("Train error:  ", 1-np.sum(labels_train == np.array(prediction_train))/len(labels_train))
    prediction_test = predict(w, X_test)
    prediction_test = np.argmax(prediction_test, axis=0)
    print("Test error:   ", 1-np.sum(labels_test == np.array(prediction_test))/len(labels_test))