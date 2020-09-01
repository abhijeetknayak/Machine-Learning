import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.step_size = 0.01
        self.eps = 1e-5

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        N, d = X.shape
        if self.theta is None:
            self.theta = np.zeros(d)

        for idx in range(100000):
            theta = self.theta.copy()
            scores = X.dot(theta)
            loss = np.mean((scores - y) ** 2)

            first_diff = (1 / N) * X.T.dot(scores - y)

            hess = (1 / N) * X.T.dot(X)
            hess_inv = np.linalg.inv(hess)

            self.theta -= 0.01 * hess_inv.dot(first_diff)
            diff = np.sum(abs(theta - self.theta))

            print("Iteration {}; Loss : {}; Change in Theta : {}".format(idx, loss, diff))

            if diff < self.eps:
                break

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        out = X.copy()
        for idx in range(2, k + 1):
            arr = X[:, 1] ** idx
            arr = arr[:, np.newaxis]
            out = np.hstack((out, arr))

        return out

        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        return X.dot(self.theta)

        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        x_ = model.create_poly(k, train_x)

        model.fit(x_, train_y)

        plot_x_mod = model.create_poly(k, plot_x)
        plot_y = model.predict(plot_x_mod)

        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(train_path)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
