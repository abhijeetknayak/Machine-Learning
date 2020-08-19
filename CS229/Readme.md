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
  -> *Using Normal Equations to learn parameters in a single iteration* : 

