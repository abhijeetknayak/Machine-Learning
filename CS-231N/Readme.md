Lecture 1:



Lecture 2:



Lecture 3:



Lecture 4:



Lecture 5:



Lecture 6: Training Networks
  1. Choice of Activation functions for your network:
      | Activation Function | f(x) | Pros | Cons |
      | :---: | :---: | :---: | :---: |
      | Sigmoid | 1 / (1 + e<sup>-x</sup>) | None. Don't use this | 1. Output isn't zero centered. <br>
      2. Kills gradient flow as a result of saturation at a high or low x-value |
      | Tanh Activation function | tanh(x) | Zero Centered output  | b |
      | Rectified Linear Unit (RELU) | max(0, x) | x | n |
      | Leaky RELU | max(0.01x, x) | x | n |
      | ELU | max(αx, x) | x | n |
  
  
  2. Data Pre-processing(Zero Centered Data)
  
  
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
      
      
