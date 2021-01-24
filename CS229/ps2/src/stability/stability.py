# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    loss = (1/2 * X.shape[0]) * np.sum((Y - probs) ** 2)
    # print(loss)
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, save_path):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 1.0

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished {} iterations; Grad : {}'.format(i, grad))
        if i % 100000 == 0:
            util.plot(X, Y, theta, str(i) + save_path)
            learning_rate /= 5
        if i % 100000 == 0:
            util.plot(X, Y, theta, str(i) + save_path)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    util.plot(X, Y, theta, save_path)
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya, 'a.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, 'b.png')

    # util.plot(Xb, Yb, np.zeros(3), 'b.png')


if __name__ == '__main__':
    main()
