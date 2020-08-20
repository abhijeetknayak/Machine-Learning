**[CS229](http://cs229.stanford.edu/) : Machine Learning** <br><br>
**Lecture 1 : Introduction to ML** <br><br>
*Supervised Learning* : Learn mapping from n-Dimensional input to a single output. For Classification problems, the output is a label(discrete). For regression problems, the output is continuous. <br>
*Unsupervised Learning* : Just input data provided, no labels. Learn from data and derive hidden information from clusters of data. <br><br>
**Lecture 2 : Linear regression** <br>
  1. Estimating a continuous value when inputs and corresponding target values are provided as training data <br>
  2. Optimize parameters such that it minimizes the cost function <img src="https://i.upmath.me/svg/J(%5Ctheta)" alt="J(\theta)" /><br>
  **Batch Gradient Descent** form is as follows. Repeat until Convergence!<br>
    <img src="https://i.upmath.me/svg/%5Ctheta_j%20%3D%20%5Ctheta_j%20-%20%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B(y%5Ep%20-%20y%5Ei)%20x%5Ei_j%7D%3B%5C%20%5Calpha%20-Learning%5C%20rate%3By%5Ep%20-Predicted%5C%20Output" alt="\theta_j = \theta_j - \alpha\sum_{i=1}^{m}{(y^p - y^i) x^i_j};\ \alpha -Learning\ rate;y^p -Predicted\ Output" /> <br>
    The main disadvantage of this is that for every update of your parameters, the algorithm needs to process the entire dataset. This might not be computationally feasible! <br>
  **Stochastic Gradient Descent** helps in these situations. All parameters are updates based the optimization of the loss function for every example. SGD takes a noisy route to the global minima, but on an average the optimization is performed in the direction of steepest gradient descent. SGD never converges though. The algorithm always tries to do better on the example under consideration. <br>
  *Using Normal Equations to learn parameters in a single iteration* :<br> After derivations, optimal values for the parameters can be reached within a single iteration. <br>
    <img src="https://i.upmath.me/svg/%5Ctheta%20%3D%20(X%5ETX)%5E%7B-1%7DX%5ETY" alt="\theta = (X^TX)^{-1}X^TY" />  , with the assumption that the matrix is invertible. <br><br>

**Lecture 3 : Locally Weighted Regression + Logistic Regression** <br>
* Feature selection is important to decide if linear equations are good enough with the data that you have, or if you would have to depend on other features(Eg x<sup>2</sup>, log(x), etc). 
* What could also be used is the Locally Weighted Regression algortihm, which weights the losses from training examples according to their proximity to the input under consideration. It is explained below :<br>

**Locally Weighted Regression** : <br>
* In cases where you training data has features which are not distributed linearly, training a linear regression model can lead to underfitting your training data.<br>
* Instead of linear regression, locally weighted regression helps weighting the losses from training examples that are near the input for which a prediction has to be made. <br>
* The loss function is defined as : <br>
    <img src="https://i.upmath.me/svg/J(%5Ctheta)%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw%5Ei(y%5Ep%20-%20y%5Ei)%7D%5C%20where%5C%20y%5Ep%20%3D%20h_%5Ctheta(x%5Ei)%3B%20w%5Ei%20%3Dweight%5C%20for%5C%20x%5Ei%20" alt="J(\theta) = \sum_{i=1}^{n}{w^i(y^p - y^i)}\ where\ y^p = h_\theta(x^i); w^i =weight\ for\ x^i " />
    <img src="https://i.upmath.me/svg/w%5Ei%3D%5Cexp%20(-%7B(x%5Ei%20-%20x)%5E2%20%5Cover%202%5Ctau%5E2%7D)%5C%20where%5C%20x%20-%20input%3B%20%5Ctau%20-%20bandwidth" alt="w^i=\exp (-{(x^i - x)^2 \over 2\tau^2})\ where\ x - input; \tau - bandwidth" />
* As new parameters are defined for every value of the input x for which we want to predict an output, all the training data needs to be stored at all points of time. This is why this is termed as a **Non-Parametric Learning Algorithm** <br>

**Probabilistic Interpretation** : <br>
**Logistic Regression** : For classification problems <br>


      
