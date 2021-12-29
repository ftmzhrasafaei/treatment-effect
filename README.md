# Evaluation of Treatment Effect
Evaluation of Treatment Effect with Sequential Neural Network 

This network uses data of 116 patients to predict two values of "Two years follow up" and "outcome".

To solve this problem, I used two sequential networks, one as a classifier and one as a regression.
I used 90% of the data for training, and 10% for validation of the models. 
- I used "Keras" module for designing networks.

### Classifier

For this network, I first used a Flatten layer. Then I used a Dense layer with 150 neurons with the ReLU activation function. In the last layer, I used the Dense layer with two output neurons and the SoftMax activation function, which determines the output probability of 0.1. This model has an accuracy of more than 90% on training data and 90 to 100% on assessment data.
In this model, I used the adam optimizer, which gives good output with low data and has a variable learning rate that improves the learning process. The error model sparse_categorical_crossentropy was also chosen, which works well for classification with natural numbers like this (0,1).

![image](https://user-images.githubusercontent.com/47606879/147673480-4d8caa6c-c6f2-402a-a283-a305560aa921.png)

![image](https://user-images.githubusercontent.com/47606879/147673497-20d76059-3703-423f-8186-d810199c3077.png)


### Regression
For this network, I used two hidden Dense layers with 500 and 50 neurons, respectively. In the last layer, I used a Dense layer with an output that shows the amount of regression. The function of activating all layers is ReLU. For error and optimizer, I used the minimum squared error and the root mean square squares, respectively.
This layer has a minimum square error of about 50 on training data and about 30 on evaluation data.

![image](https://user-images.githubusercontent.com/47606879/147673598-a9270353-04a2-447e-b264-6bd3eaa1775e.png)


