import numpy as np
import util
from sklearn.linear_model import LogisticRegression


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***

    # Check Data size
    # print(x_train.shape, y_train.shape)

    # Create instance of logistic Regression Model
    model = LogisticRegression()

    # Fit model to training data
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, './', )

    # Predict values for Validation Set
    # model.predict(x_val)

    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Train model until convergence
        N, dim = x.shape

        if self.theta is None:
            self.theta = np.zeros(dim)

        for idx in range(self.max_iter):
            theta = self.theta
            h_x = x.dot(theta)
            scores = 1 / (1 + np.exp(-h_x))  # Broadcast operations
            # loss = (-1 / N) * np.sum(y * np.log(scores) + (1 - y) * np.log(1 - scores))

            # First Differential
            first_diff = (-1 / N) * x.T.dot(y - scores)  # [D, ]

            # Find the Hessian matrix - x'.D.x, where D is a diagonal matrix sigm(x)*(1 - sigm(x))
            hess = (1 / N) * x.T.dot(np.diag(scores * (1 - scores))).dot(x)  # [D, D]

            # Update theta values every iteration
            self.theta = theta - np.linalg.inv(hess).dot(first_diff)  # [D * D].dot([D, ])
            diff = np.sum(abs(self.theta - theta))
            if self.verbose:
                print("Iteration {}; Change in theta : {}".format(idx, diff))

            # Norm should be less than the threshold. If it is, end training
            if diff < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
