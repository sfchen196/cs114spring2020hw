import numpy as np
from scipy.special import expit

#                    x_Chinese x_Tokyo x_Japan dummy_feature
x_train = np.array([[2,        0,      0,      1],
                    [2,        0,      0,      1],
                    [1,        0,      0,      1],
                    [1,        1,      1,      1]])
#                    1 for Chinese, 0 for Japanese
y_train = np.array([1, 1, 1, 0])

x_test = np.array([[3, 1, 1, 1]])
y_test = np.array([1])

class LogisticRegression():

    def __init__(self):
        self.theta = None

    def train(self, x_train, y_train, eta=0.1, epochs=10):
    #                          w_Chinese w_Tokyo w_Japan b
        self.theta = np.array([0.,       0.,     0.,     0.])
        #              how many passes through the training data
        for i in range(epochs):
            order = list(range(len(y_train)))
            #np.random.shuffle(order)
            cost = 0.
            for j in order:
                x = x_train[j]
                y = y_train[j]
                y_hat = expit(np.dot(self.theta, x))
                cost -= (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
                gradient = (y_hat - y) * x
                self.theta -= eta * gradient
            cost /= len(y_train)
            print('after epoch',i+1, ', cost =',cost, ' and theta =',self.theta)

    def test(self, x_test, y_test):
        accuracy = 0.
        for j in range(len(y_test)):
            x = x_test[j]
            y = y_test[j]
            y_hat = expit(np.dot(self.theta, x))
            if np.round(y_hat) == y:
                accuracy += 1
        accuracy /= len(y_test)
        print('accuracy =', accuracy, ' and y_hat =', y_hat)
