import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()

    # Fit model to the training data. Define theta
    model.fit(x_train, y_train)

    # Read validation set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    # Save predictions to save path
    np.savetxt(save_path, model.predict(x_val))

    # Plot boundaries
    util.plot(x_val, y_val, model.theta, save_path[:-4])

    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        N = x.shape[0]
        y_1 = (y == 1)
        y_0 = (y == 0)

        phi = np.mean(y_1)
        mu_0 = np.sum(x[y == 0], axis=0) / np.sum(y_0)
        mu_1 = np.sum(x[y == 1], axis=0) / np.sum(y_1)

        A = x
        A[y == 0] -= mu_0
        A[y == 1] -= mu_1

        sigma = A.T.dot(A) / N
        sigma_inv = np.linalg.inv(sigma)

        # Write theta in terms of the parameters
        if self.theta is None:
            self.theta = np.zeros(x.shape[1] + 1)
        self.theta[0] = -np.log((1 - phi) / phi) - 0.5 * ((mu_1.T.dot(sigma_inv)).dot(mu_1)
                                                    - (mu_0.T.dot(sigma_inv)).dot(mu_0))
        self.theta[1:] = mu_1.T.dot(sigma_inv) - mu_0.T.dot(sigma_inv)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # validation set has intercept term added
        scores = x.dot(self.theta)

        # Sigmoid with theta as parameter
        scores_sig = 1 / (1 + np.exp(-scores))

        # Round off scores to nearest integer
        y_pred = np.round(scores_sig)

        return y_pred

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
