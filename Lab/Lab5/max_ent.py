import numpy as np

npzfile = np.load('mnist.npz')
x_train = npzfile['X_train'].reshape(60000, 784) / 255
y_train = npzfile['y_train']
x_test = npzfile['X_test'].reshape(10000, 784) / 255
y_test = npzfile['y_test']

def softmax(x):
    stable_x = np.exp(x - np.max(x))
    return stable_x / np.sum(stable_x, axis=0)

class MaxEnt():

    def __init__(self):
        self.theta = None

    def train(self, x_train, y_train, eta=0.01, epochs=10):
        num_classes = np.max(y_train) + 1
        num_features = len(x_train[0])
        self.theta = np.random.rand(num_classes, num_features + 1) - 0.5
        for i in range(epochs):
            order = list(range(len(y_train)))
            np.random.shuffle(order)
            cost = 0.
            accuracy = 0.
            for j in order:
                x = np.zeros(num_features + 1)
                x[:-1] = x_train[j]
                x[-1] = 1
                y = np.zeros((num_classes, 1))
                y[y_train[j]] = 1
                y_hat = softmax(np.dot(self.theta, x))[:, None]
                if np.argmax(y_hat) == y_train[j]:
                    accuracy += 1
                cost -= np.sum(y * np.log(y_hat))
                gradient = (y_hat - y) * x
                self.theta -= eta * gradient
            cost /= len(x_train)
            accuracy /= len(x_train)
            print('after epoch',i+1, ', cost =',cost, 'and accuracy =',accuracy)

    def test(self, x_test, y_test):
        num_classes = len(self.theta)
        num_features = len(self.theta[0]) - 1
        accuracy = 0.
        for j in range(len(y_test)):
            x = np.zeros(num_features + 1)
            x[:-1] = x_test[j]
            x[-1] = 1
            y_hat = softmax(np.dot(self.theta, x))[:, None]
            if np.argmax(y_hat) == y_test[j]:
                accuracy += 1
        accuracy /= len(x_test)
        print('accuracy =', accuracy)
