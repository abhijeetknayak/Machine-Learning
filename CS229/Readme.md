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
  * EDIT
  * Maximum log likelihood estimate on the joint likelihood<br>
  
## **Lecture 6 : Spam Classifiers with Laplace Smoothing + Support Vector Machines** <br>
* Spam Classifier : <br>
  * For all the words in an email, create a one-hot encoded vector representation such that if a word from the vocabulary is available in the email, set it to 1(0 otherwise).
  This way you would have a (N * D) matrix, where N is the number of training examples, D is the number of words in the vocabulary. This is called as the **Multivariate Bernoulli Event Model** <br>
  * Use **Gaussian Discriminant Analysis** and **Naive-Bayes**, to estimate p(y|x) using p(x|y) and p(y) as follows : <br>
  Formulae <br>
  * This spam classifier has problems with unseen words in the new examples. Probabilities of seeing the words in an email would be zero, thereby causing the probability of classifying an email as spam to be undefined (0/0) <br>
  * **Laplace Smoothing**(as shown below) helps here, by augmenting the numerator and denominator with 1's, so as to enable numerically stable computation and to make sure that the probabilities aren't hard coded to zero in such cases.  <br>
  * A new representation can also be used. In **Multinomial Event Model** ----> Explain
* **Support Vector Machine(SVM)** : <br>
  * If the data from the training set cannot be classified using a linear classifier, a feature map needs to be created that can learn a non-linear decision boundary to differentiate between samples<br>
  * The problem here is that **defining the feature map** is hard to do. SVMs helps in these problems, by defining complex, higher dimensional feature maps and using a linear classifier to classify samples of data <br> 
  * SVM Notation : 'Y' values can be {-1, 1} (and not {0, 1} ). The bias term is now separated. Therefore  <img src="https://i.upmath.me/svg/%5Ctheta%5ETX%20%3D%20W%5ETX%20%2B%20b" alt="\theta^TX = W^TX + b" /> <br>
  * **Functional Margin** defines the values of the scores where a transition from positive to negative predictions happens. FM of a hyperplane is given by :
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/%5Chat%20%5Cgamma%5Ei%20%3D%20y%5Ei%20*%20(W%5ETX%20%2B%20b)" alt="\hat \gamma^i = y^i * (W^TX + b)" /><br>
  If y<sup>i</sup> = 1, corresponding score needs to be >> 0. Similarly, if y<sup>i</sup> = -1, scores needs to be << 0 for a good functional margin. Additionally, if a training example has an **FM > 0**, it means that the example is classified correctly. <br>
  FM for the entire training set would be the **minimum value of the FMs for all training examples(worst case)**
  * **Geometric Margin** defines the separation between the linear classifier and samples in the data. A classifier with higher geomtric margin is preferred.
  The GM for an example is given by : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/%5Cgamma%5Ei%20%3D%20%7By%5Ei%20*%20(W%5ETX%20%2B%20b)%20%5Cover%20%7C%7CW%7C%7C%7D" alt="\gamma^i = {y^i * (W^TX + b) \over ||W||}" /><br>
  ||W|| is the euclidean norm of W<br>
  GM for the training data is the minimum GM out of all training examples <br>
  * **Optimal Margin Classifier** : Define parameters (W, b) to maximize the GM. <br>

## **Lecture 7 : Kernels**
* Optimization for the Optimal Margin Classifier can be represented as :
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/min%20%5C%20%7B1%20%5Cover%202%7D%20%7C%7CW%7C%7C%5E2%20%5C%20subject%5C%20to%5C%20y%5Ei(W%5ETX%5Ei%20%2B%20b)%20%3E%3D%201" alt="min \ {1 \over 2} ||W||^2 \ subject\ to\ y^i(W^TX^i + b) &gt;= 1" />
* The parameters can be represented as a linear combination of the inputs.
  * Intuition 1 : This is because we always start optimization with zero valued parameters, and continuously add a portion of the input to the parameters.
  * Intuition 2 : Vector pertaining to the parameters is perpendicular to the decision boundary. **W** sets the direction of the decision boundary, where as **b** varies the position of the decision boundary. As **W** spans the vector space, we can assume that the parameter is a linear combination of the inputs.
  * Using this property, an efficient **O(n)** computation can be used to update parameters -----> Kernel Trick.
* **Kernel Trick** : 
  * Start by finding the inner product of the feature map of X with itself. This is called as the Kernel.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/K(X%2C%20Z)%20%3D%5C%20%20%3C%5Cphi(X)%2C%20%5Cphi(Z)%3E%20%5C%20%3D%20%5Cphi(X)%5ET%5Cphi(Z)" alt="K(X, Z) =\  &lt;\phi(X), \phi(Z)&gt; \ = \phi(X)^T\phi(Z)" /> <br>
  * Replace inner products <X, Z> with K(X, Z). Finding K(X, Z) is an efficient manner helps solve the problem of having a large number of dimensions in the feature map.
  * **The Kernel function that is used depends on the feature mapping** <img src="https://i.upmath.me/svg/X%20%5C%20-%3E%20%5Cphi(X)" alt="X \ -&gt; \phi(X)" />
  * **Linear Kernel** : The high dimensional feature map is the same as the input vectors.
  * **Polynomial Kernel** : &nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/K(X%2C%20Z)%20%3D%5C%20%20(X%5ETZ%20%2B%20C)%5Ed" alt="K(X, Z) =\  (X^TZ + C)^d" />
  * **Gaussian Kernel** : &nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/K(X%2C%20Z)%20%3D%20%5Cexp(-%20%7B%7C%7CX%20-%20Z%7C%7C%5E2%20%5Cover%202%5Csigma%5E2%7D)" alt="K(X, Z) = \exp(- {||X - Z||^2 \over 2\sigma^2})" />
* **Support Vector Machines** : Applying the Kernel trick on an optimal margin classifier ---> SVM. This allows us to use a high dimensional feature space, but computational complexity still remains O(n)
* How to define a Kernel :
  * **Mercer's Theorem** : K is a valid Kernel function, if there exists a valid function <img src="https://i.upmath.me/svg/%5Cphi(X)" alt="\phi(X)" /> such that <img src="https://i.upmath.me/svg/K(X%2C%20Z)%20%3D%20%5Cphi(X)%5ET%5Cphi(Z)" alt="K(X, Z) = \phi(X)^T\phi(Z)" />, if and only if for any 'd' points, the corresponding Kernel matrix K is Positive semi-definite
* **L1 Norm Soft Margin SVM** : To prevent overfitting while using Kernels, a soft margin can be used during classification. The optimization function now changes to :
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.upmath.me/svg/min%20%5C%20%7B1%20%5Cover%202%7D%20%7C%7CW%7C%7C%5E2%20%2B%20C%5Csum_%7Bi%3D1%7D%5En%7B%5Cxi_i%7D%20%0A%5C%5C%20subject%5C%20to%5C%20y%5Ei(W%5ETX%5Ei%20%2B%20b)%20%3E%3D%201%20-%20%5Cxi_i%5C%20%3B%20%5C%20%5Cxi_i%20%3E%3D%200" alt="min \ {1 \over 2} ||W||^2 + C\sum_{i=1}^n{\xi_i} 
\\ subject\ to\ y^i(W^TX^i + b) &gt;= 1 - \xi_i\ ; \ \xi_i &gt;= 0" />

## **Lecture 8 : Bias, Variance**
* Bias

* Variance

* Cross Validation techniques :
  * (Simple) Hold-out Cross Validation : Divide data into train, validation and test sets. Always validate the model using Vlaidation set. Thereafter, when the model is ready for a test, run predictions on the test set
  * K-fold Cross Validation : Divide the training data into 'k' pieces. Train on 'k - 1' pieces of data and validate on the last piece. Repeat this process using all pieces as the valiudation set and then average the results obtained
  * Leave-one-out Cross Validation : Leave out one sample as the validation 'set'. Train on all other samples, validate on one sample
* Feature Selection

## **Lecture 9 : Bias, Variance Continued**
* Some assumptions : All data used in the train and test sets are sampled from the same data generating distribution. All of these data points are independant of each other
* Data View of Bias and Variance
* Parameter View : <br>
<img src="https://github.com/abhijeetknayak/Machine-Learning/blob/master/CS229/Material/Bias-Variance.png" /> <br>
* **Reducing Variance : Regularization** helps in reducing the variance, but adds a small bias to the learning model <br>
* **Reducing Bias :** Increase hypothesis space. This reduces the bias because the learning algorithm is not biased to a certain set of algorithms. This might increase the variance though<br>
* Bias-Variance Tradeoff :<br>
<img src="https://github.com/abhijeetknayak/Machine-Learning/blob/master/CS229/Material/Bias-Variance-Tradeoff.JPG" />

  
  
  
  
  
  
  
  
  
  
  
  
  
