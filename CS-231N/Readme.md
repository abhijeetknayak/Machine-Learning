**Lecture 1** : 
An overview of Computer Vision and problems everyone is trying to solve.

**Lecture 2 : K-Nearest Neighbor Classifier**


**Lecture 3 : **



**Lecture 4 : Loss Functions and Optimization**



**Lecture 5 : Convolutional Neural Networks**



**Lecture 6 : Training Networks**
  1. Choice of Activation functions for your network:
      | Activation Function | f(x) <img></img> | Pros | Cons |
      | :---: |---| :--- | :--- |
      | Sigmoid | 1 / (1 + e<sup>-x</sup>) | -> None. Don't use this | -> Output isn't zero centered. <br> -> Kills gradient flow as a result of saturation at a high or low x-value |
      | Tanh Activation function | tanh(x) | -> Zero Centered output. Optimization is efficient compared to sigmoid <br>  | -> Saturation kills the gradient flow(high or low x-value) |
      | Rectified Linear Unit (RELU) | max(0, x) | -> No saturation in the positive region. <br> -> Efficient computation and faster convergence | -> Non zero centered. <br> -> Dead Relu's if x < 0 (Can happen because of bad initialization of weights or if the learning rate is too high) <br>|
      | Leaky RELU | max(0.01x, x) | -> All pros of Relu's <br> -> No dying Relu's here because of a gradient in the negative direction too. | n |
      | Parametric RELU | max(αx, x) | α is a hyperparameter that needs to be optimized | -> |
      | Exponential LU | = x  if x > 0 <br> = α * (e<sup>x</sup> - 1) if x <= 0 | -> All RELU benfits hold for this. <br> -> Almost zero centered output. <br> -> Negative region saturates for a large negative value. Provides Noise robustness.| -> |
      |<img width=200/>|<img width=375/>| | |
  
  
  2. Data Pre-processing(Zero Centered Data) :
      Why is zero mean data required? Assume a scenario where all the inputs(X) is positive. You forward propagate through the network and obtain a loss value.<br>
      During backprop at this layer, the upstream gradient can either be positive or negative. The local gradient of W is X.<br>
      Therefore, the gradient of W is either all positive or all negative. This leaves us with only two directions of optimizations, which may not be optimal.<br>
      By zero centering your data, you would have an equal number of poisitve and negative inputs(almost), thereby providing multiple directions for updates.<br>
      
      **X_new = (X - Mean(X)) / StdDev(X)**
  
  
  3. Initialization of Weights for robustness of network
  
  
  4. Batch Normalization to prevent dead units in your network
  
  
  5. Training Process :
      1. Always start your training process with a small subset of the data from your training set.
      2. Start with Zero Regularization and ensure that the value for the loss(or cost) is plausible.
      3. Update the learning rate to a small value(say 1e-6) and check how this affects the network loss.
      4. Similarly try out a small value for the regularization and check how this affects the loss. You should expect a slight increase in loss.
      5. Optimization in the log space is easier. Use the random.uniform function in the numpy library(10**uniform(l, h)).
      6. A random search for these hyper-parameters works out better than the grid search.
      7. Look at all the training and Validation accuracies achieved by the network and try to optimize in areas of high accuracy.      
      8. Monitor plots of training, validation accuracies as well as loss curves:          
         A big difference between Train and Val accuracy(Overfitting) - Increase Regularization; Small difference - Model saturated. Increase model capacity.
         Monitor loss curves for information about the learning rate effectiveness.
         
**Lecture 7 : Optimization Algorithms** <br><br>
  A Number of optimization methods are defined. The simplest one among these is the Stochastic Gradient descent. Although its a good way to optimize the parameters of your model, the algorithm has its own drawbacks, which leads to inefficient optimization.<br>
  These are : <br>
  1. If the sensitivity of the parameters are different along different directions, it results in sub-optimal optimization(zig-zag) instead of going directly along the direction of highest gradient descent.
  2. Stochasticity introduces its own problems, where in the mini-batch that is used to update the parameters in an iteration might lead to noisy gradients, thereby resulting in a wrong update. The updates, thereby, are jittery.
  3. Sub-optimal minima or Saddle points may be reached by the function. In regions around these saddle points, the gradients are close to zero, which would reduce the rate of update and inturn, increase the training time. 
  
  Inorder to resolve these issues with the optimization algorithm, a number of other techniques are defined : <br>
   1. **SGD with Momentum** : This is similar to the SGD but parameters are updated, not using the gradient, but with a 'velocity' value. This 'velocity' value is defined by using the gradient. The accumulated velocity is used as the gradient. The intuition here is that if you reach a saddle point, the momentum accumulated would help you cross the saddle point. <br>
       ----> **V<sub>t+1</sub> = beta * V<sub>t</sub> + grad** <br>
       ----> **X -= learning_rate * V<sub>t+1</sub>**
   2. **AdaGrad Optimizer** : Here, a Gradient squared term is accumulated with the square of the gradient. This is used later during optimization.
    The intuition here is that if the gradient along a particular direction is large, the denominator in the formula would be large too, thereby slowing down the update along that direction. <br>
       ----> **gradSquared += grad * grad**
       ----> **X -= learning_rate * grad / (sqrt(gradSquared) + eps)**  
   3. **RMS Prop** : This is similar to AdaGrad Optimizer. Here, the gradSquared term is decayed using a decay_rate.
   Decay Rate is a hyperparameter(0.9 or 0.99) <br>
       ----> **gradSquared = decay_rate * grdaSquared + (1 - decay_rate) * grad * grad** <br>
       ----> **X -= learning_rate * grad / (sqrt(gradSquared) + eps)**   
   4. **Adam Optimizer** : Adam Optimizer combines the best properties of SGD with momentum and RMS Prop. 
       ----> **f_m, s_m = 0, 0** <br>
       ----> **f_m = beta1 * f_m + (1 - beta1) * grad** (Momentum term)<br>
       ----> **s_m = beta2 * s_m + (1 - beta2) * grad * grad** (Grad Squared Term)<br>
       ----> **X -= learning_rate * f_m / sqrt(s_m) + eps**<br>
       But there's a problem here. If we start with **s_m = 0**, during the first iteration the denominator **sqrt(s_m) + eps** is very small. This would increase the rate of update. We would be taking a very large step during the first few iterations as a result of this bias.<br>
       Therefore, bias correction is defined :<br>
       ----> **f_b = f_m / (1 - beta1 ^ t)**<br>
       ----> **s_b = s_m / (1 - beta2 ^ t)**<br>
       ----> **X -= learning_rate * f_b / sqrt(s_b) + eps**<br>   
    
  In addition to these optimization algorithms, some learning rate decay techniques are also defined which would be helpful if the loss curves tend to plateau after a few iterations of training. <br>
    1. Exponential Decay : **α = α<sub>0</sub> * e<sup>-Kt</sup>** (K - Decay rate)<br>
    2. 1 / t Decay : **α = α<sub>0</sub> / (1 + Kt)** (K - Decay rate) <br>
    3. Decay in phases : Depnding on the total number of iterations, decay the learning rate after a certain count is reached.
    
    
    
  
  

      
      
