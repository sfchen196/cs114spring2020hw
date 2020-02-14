import numpy as np
from scipy.special import expit

npzfile = np.load('mnist.npz')
x_train = npzfile['X_train'].reshape(60000, 784) / 255
y_train = npzfile['y_train']
x_test = npzfile['X_test'].reshape(10000, 784) / 255
y_test = npzfile['y_test']

def softmax(x):
    stable_x = np.exp(x - np.max(x))
    return stable_x / np.sum(stable_x, axis=0)

class MLP():

    def __init__(self, num_hidden=300):
        self.num_hidden = num_hidden
        self.hidden_weights = None
        self.output_weights = None

    def train(self, x_train, y_train, eta=0.01, epochs=10):
        num_classes = np.max(y_train) + 1
        num_features = len(x_train[0])
        self.hidden_weights = np.random.rand(self.num_hidden,num_features+1)-0.5
        self.output_weights = np.random.rand(num_classes, self.num_hidden+1)-0.5
        for i in range(epochs):
            order = list(range(len(y_train)))
            np.random.shuffle(order)
            cost = 0.
            accuracy = 0.
            for j in order:
                x = np.zeros(num_features + 1)
                x[:-1] = x_train[j]
                x[-1] = 1
                hidden_output = np.zeros(self.num_hidden + 1)
                hidden_output[:-1] = expit(np.dot(self.hidden_weights, x))
                hidden_output[-1] = 1
                y = np.zeros((num_classes, 1))
                y[y_train[j]] = 1
                y_hat=softmax(np.dot(self.output_weights,hidden_output))[:,None]
                if np.argmax(y_hat) == y_train[j]:
                    accuracy += 1
                cost -= np.sum(y * np.log(y_hat))
                output_gradient = (y_hat - y) * hidden_output
                hidden_gradient=(np.dot(self.output_weights[:,:-1].T,(y_hat-y))*
                                 hidden_output[:-1, None] *
                                 (1 - hidden_output[:-1, None]) *
                                 x)
                self.output_weights -= eta * output_gradient
                self.hidden_weights -= eta * hidden_gradient
            cost /= len(x_train)
            accuracy /= len(x_train)
            print('after epoch',i+1, ', cost =',cost, 'and accuracy =',accuracy)

    def test(self, x_test, y_test):
        num_classes = len(self.output_weights)
        num_features = len(self.hidden_weights[0]) - 1
        accuracy = 0.
        for i in range(len(y_test)):
            x = np.zeros(num_features + 1)
            x[:-1] = x_test[i]
            x[-1] = 1
            hidden_output = np.zeros(self.num_hidden + 1)
            hidden_output[:-1] = expit(np.dot(self.hidden_weights, x))
            hidden_output[-1] = 1
            y_hat = softmax(np.dot(self.output_weights, hidden_output))[:, None]
            if np.argmax(y_hat) == y_test[i]:
                accuracy += 1
        accuracy /= len(x_test)
        print('accuracy =', accuracy)
