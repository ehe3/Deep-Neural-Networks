# Deep-Neural-Networks
A repository for implementing deep-learning tools from scratch.

## dnn.py
A 3-layer feedforward deep neural net that utilizes a leaky-relu non-linearity and a soft-max classifier.
The architure of the DNN follows (input layer) -> (hidden layer) -> (leaky-relu) -> (hidden layer) -> (leaky-relu) -> (hidden layer) -> (soft-max).

200,000 iterations of mini-batch SGD with a batch-size of 64, a learning rate of 0.0001, an input layer size of 784 (MNIST image size), a second layer size of 100, and a third layer size of 25 on the MNIST training set resulted in a classification error of 9.18%. Alpha for leaky-relu was set to 0.1 and weights and biases were initialized by sampling from a normal distribution with mean 0 and variance 0.1. Training time was approximately 30 minutes.
