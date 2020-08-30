**[CS229](http://cs229.stanford.edu/) : Machine Learning** <br>
## **Lecture 1 : Introduction to ML** <br>
* Supervised Learning : Learn mapping from n-Dimensional input to a single output. For Classification problems, the output is a label(discrete). For regression problems, the output is continuous. <br>
* Unsupervised Learning : Just input data provided, no labels. Learn from data and derive hidden information from clusters of data. <br>

## **Lecture 2 : Linear regression** <br>
  1. Estimating a continuous value when inputs and corresponding target values are provided as training data <br>
  2. Optimize parameters such that it minimizes the cost function <img src="https://i.upmath.me/svg/J(%5Ctheta)" alt="J(\theta)" /><br>
  **Batch Gradient Descent** form is as follows. Repeat until Convergence!<br>
    <img src="https://i.upmath.me/svg/%5Ctheta_j%20%3D%20%5Ctheta_j%20-%20%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B(y%5Ep%20-%20y%5Ei)%20x%5Ei_j%7D%3B%5C%20%5Calpha%20-Learning%5C%20rate%3By%5Ep%20-Predicted%5C%20Output" alt="\theta_j = \theta_j - \alpha\sum_{i=1}^{m}{(y^p - y^i) x^i_j};\ \alpha -Learning\ rate;y^p -Predicted\ Output" /> <br>
    The main disadvantage of this is that for every update of your parameters, the algorithm needs to process the entire dataset. This might not be computationally feasible! <br>
  **Stochastic Gradient Descent** helps in these situations. All parameters are updates based the optimization of the loss function for every example. SGD takes a noisy route to the global minima, but on an average the optimization is performed in the direction of steepest gradient descent. SGD never converges though. The algorithm always tries to do better on the example under consideration. <br>
  *Using Normal Equations to learn parameters in a single iteration* :<br> After derivations, optimal values for the parameters can be reached within a single iteration. <br>
    <img src="https://i.upmath.me/svg/%5Ctheta%20%3D%20(X%5ETX)%5E%7B-1%7DX%5ETY" alt="\theta = (X^TX)^{-1}X^TY" />  , with the assumption that the matrix is invertible. <br><br>

## **Lecture 3 : Locally Weighted Regression + Logistic Regression** <br>
* Feature selection is important to decide if linear equations are good enough with the data that you have, or if you would have to depend on other features(Eg x<sup>2</sup>, log(x), etc). 
* What could also be used is the Locally Weighted Regression algortihm, which weights the losses from training examples according to their proximity to the input under consideration. It is explained below :<br>

**Locally Weighted Regression** :
* In cases where you training data has features which are not distributed linearly, training a linear regression model can lead to underfitting your training data.<br>
* Instead of linear regression, locally weighted regression helps weighting the losses from training examples that are near the input for which a prediction has to be made. <br>
* The loss function is defined as : <br>
    <img src="https://i.upmath.me/svg/J(%5Ctheta)%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw%5Ei(y%5Ep%20-%20y%5Ei)%7D%5C%20where%5C%20y%5Ep%20%3D%20h_%5Ctheta(x%5Ei)%3B%20w%5Ei%20%3Dweight%5C%20for%5C%20x%5Ei%20" alt="J(\theta) = \sum_{i=1}^{n}{w^i(y^p - y^i)}\ where\ y^p = h_\theta(x^i); w^i =weight\ for\ x^i " /><br>
    <img src="https://i.upmath.me/svg/w%5Ei%3D%5Cexp%20(-%7B(x%5Ei%20-%20x)%5E2%20%5Cover%202%5Ctau%5E2%7D)%5C%20where%5C%20x%20-%20input%3B%20%5Ctau%20-%20bandwidth" alt="w^i=\exp (-{(x^i - x)^2 \over 2\tau^2})\ where\ x - input; \tau - bandwidth" />
* As new parameters are defined for every value of the input x for which we want to predict an output, all the training data needs to be stored at all points of time. This is why this is termed as a **Non-Parametric Learning Algorithm** <br>

**Probabilistic Interpretation** :
* Start with an assumption of Independently and Identically distributed input data from a Normal Distribution. 
* Find the __Maximum Likelihood Estimate__ for the parameters of the model. Maximizing log likelihood makes it simpler.
* The resulting cost function(minimization) is similar to the least squares minimization in linear regression, which tells us that the least squares minimization was a natural way to optimize the parameters of the model.

**Logistic Regression** : For classification problems <br>

## **Lecture 4 : Perceptron + Exponential Family + GLMs + Multiclass Regression**<br>
* **Perceptron** : This is exactly the same as logistic regression. The only difference here is that h(x) used is different. <br>
For a perceptron, scores are fed to a function, which is the hard version of the sigmoid.
**g(z) = 1  if z >= 0; 0 if z < 0**  The updation of parameters remains exactly the same. <br>
The intuition here is that for every example which is classified correctly, the loss values are set to zero, leading to zero update on the parameters. Whenever an exmaple is classified wrongly, the parameters are updated by adding a small portion of the input to the parameter. As the parameters inch closer to the input, the dot product maximizes and hence, leads to proper classification. <br>
* **Exponential Family** : <br>
A distribution is said to be derived from the exponential family if the PDF of the distribution is of the form <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/p(y%3B%5Ceta)%20%3D%20b(y)%5Cexp(%5Ceta%5ETT(y)%20-%20a(%5Ceta))" alt="p(y;\eta) = b(y)\exp(\eta^TT(y) - a(\eta))" /> <br>
Here, <img src="https://i.upmath.me/svg/y%20-%20data%3B%20%5Ceta%20-%20parameter%3B%20b(y)%20-%20base%5C%20measure%3B%20T(y)%20-%20Sufficient%5C%20Statistic%3B%20a(%5Ceta)%20-%20log%5C%20partition" alt="y - data; \eta - parameter; b(y) - base\ measure; T(y) - Sufficient\ Statistic; a(\eta) - log\ partition" /> <br> 
Properties of exponential family : <br>
  1. Maximum Likelihood estimate is Concave, Negative log likelihood is convex.
  2. Expectation of data is the first differential of <img src="https://i.upmath.me/svg/a(%5Ceta)" alt="a(\eta)" />
  3. Variance of the data is the second differential(Hessian) of <img src="https://i.upmath.me/svg/a(%5Ceta)" alt="a(\eta)" />
* **Generalized Linear Model(GLM)** : <br>
  1. In a generalized linear model, a distribution is chosen which defines the function through which the model scores(<img src="https://i.upmath.me/svg/%5Ctheta%5ETX" alt="\theta^TX" />) are passed through <br>
  2. The distribution that is chosen depends on the application of the model. For instance, if binary values(0 or 1) are required as outputs, a **Bernoulli** distribution is chosen. If the output can be any number on the real line, a **Gaussian** Distribution is chosen. If positive integers are what you need, a **Poisson** Distribution is chosen<br>
  3. The work flow here is : Pass input through the model to get scores(by use of theta). Pass these scores(<img src="https://i.upmath.me/svg/%5Ctheta%5ETX" alt="\theta^TX" />) through the chosen distribution, which helps you define your loss function <br>
  4. The update rule, again, stays the same. The only thing which changes here is the value of <img src="https://i.upmath.me/svg/h_%5Ctheta(x)" alt="h_\theta(x)" />, which depends on the distribution that is chosen for the model<br>

* **Softmax Regression** : <br>
  1. Cross Entropy essentially is the **distance** between the target distribution and the normalized dstribution after the input is passed through the model<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/Cross%5C%20Entropy(p%2C%20%5Chat%20p)%3D-%5Csum_%7By%5C%20%5Cepsilon%5C%20c%7Dp(y)log%5C%20%5Chat%20p(y)%20%3D%20-log(%5Chat%20p_c_i)" alt="Cross\ Entropy(p, \hat p)=-\sum_{y\ \epsilon\ c}p(y)log\ \hat p(y) = -log(\hat p_c_i)" />  where c<sub>i</sub> is the correct class for that example input. <br>

## **Lecture 5 : Generative and Discriminative Models**
* Discriminative learning algorithms learn to classify new inputs. A mapping from x -> y (<img src="https://i.upmath.me/svg/p(y%7Cx)" alt="p(y|x)" />) is learnt, where in, given a new input 'x', a class can be determined by using the mapping <br>
* Generative learning algorithms learn the features pertaining to a certain class. In essence, <img src="https://i.upmath.me/svg/p(x%7Cy)" alt="p(x|y)" /> is learnt by the algorithm. The class prior <img src="https://i.upmath.me/svg/p(y)" alt="p(y)" /> is already known, and using these values, it is easy to estimate <img src="https://i.upmath.me/svg/p(y%7Cx)" alt="p(y|x)" /> using the Naive Bayes rule.
* **Gaussian Discriminant Analysis** : <br>
  * Maximum log likelihood estimate on the joint likelihood

