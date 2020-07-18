Lecture 1:



Lecture 2:



Lecture 3:



Lecture 4:



Lecture 5:



Lecture 6: Training Networks
  1. Choice of Activation functions for your network
      Table
  
  
  2. Data Pre-processing(Zero Centered Data)
  
  
  3. Initialization of Weights for robustness of network
  
  
  4. Batch Normalization to prevent dead units in your network
  
  
  5. Training Process :
      *Always start your training process with a small subset of the data from your training set.
      *Start with Zero Regularization and ensure that the value for the loss(or cost) is plausible.
      *Update the learning rate to a small value(say 1e-6) and check how this affects the network loss.
      *Similarly try out a small value for the regularization and check how this affects the loss. You should expect a slight increase in loss.
      *Optimization in the log space is easier. Use the random.uniform function in the numpy library(10**uniform(l, h)).
      *A random search for these hyper-parameters works out better than the grid search.
      *Look at all the training and Validation accuracies achieved by the network and try to optimize in areas of high accuracy.      
      *Monitor plots of training, validation accuracies as well as loss curves:          
         *A big difference between Train and Val accuracy(Overfitting) - Increase Regularization; Small difference - Model saturated. Increase model capacity.
         *Monitor loss curves for information about the learning rate effectiveness.
      
      
