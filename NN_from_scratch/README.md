# Neural Network from Scratch in TensorFlow
## Introduction

This project implements a simple neural network from scratch using TensorFlow. It includes the full process from initializing the network, performing forward propagation, computing loss, updating parameters, and training the model. The implementation demonstrates the core principles of neural networks and provides a foundation for further exploration and customization.
## Purpose

The goal of this project is to implement a Neural Network model in TensorFlow using its core functionality (i.e., without the help of a high-level API like Keras). While it’s easier to get started with TensorFlow with the Keras API, it’s still worth understanding how a slightly lower-level implementation might work in TensorFlow.
## Requirements

    Python 3.x
    TensorFlow 2.15.0
    NumPy
    Matplotlib

## Getting Started
### Importing Libraries

First, ensure you have all necessary libraries installed:


    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import nn_utils
    %matplotlib inline

    print('TensorFlow Version:', tf.__version__)

## Neural Network Implementation
### NeuralNetwork Class

The NeuralNetwork class encapsulates the entire process of building and training a neural network. Below is a summary of its methods and functionalities:

1. Initialization (__init__): Initializes the network layers and parameters.
2. Setup (setup): Applies 'He' initialization to weights and zero initialization to biases.
3. Forward Propagation (forward_prop): Computes the forward pass of the network.
4. Compute Loss (compute_loss): Calculates the loss using softmax cross-entropy.
5. Update Parameters (update_params): Updates the network parameters using gradient descent.
6. Predict (predict): Generates predictions for input data.
7. Info (info): Prints the network's architecture and parameter count.
8, Train on Batch (train_on_batch): Trains the network on a single batch of data.
9. Train (train): Trains the network over multiple epochs.

Code

```

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_classes = layers[-1]

        self.W = {}
        self.b = {}
        self.dW = {}
        self.db = {}

        self.setup()

    def setup(self):
        for i in range(1, self.L):
            self.W[i] = tf.Variable(np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2. / self.layers[i-1]), dtype = np.float32)
            self.b[i] = tf.Variable(tf.zeros(shape=(self.layers[i], 1)))

    def forward_prop(self, A):
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        for i in range(1, self.L):
            Z = tf.matmul(A, tf.transpose(self.W[i])) + tf.transpose(self.b[i])
            if i != self.L - 1:
                A = tf.nn.relu(Z)
            else:
                A = Z
        return A

    def compute_loss(self, A, Y):
        loss = tf.nn.softmax_cross_entropy_with_logits(Y, A)
        return tf.reduce_mean(loss)

    def update_params(self, lr):
        for i in range(1, self.L):
            self.W[i].assign_sub(lr * self.dW[i])
            self.b[i].assign_sub(lr * self.db[i])

    def predict(self, X):
        A = self.forward_prop(X)
        return tf.argmax(tf.nn.softmax(A), axis=1)

    def info(self):
        num_params = 0
        for i in range(1, self.L):
            num_params += self.W[i].shape[0] * self.W[i].shape[1]
            num_params += self.b[i].shape[0]
        print('Input Features:', self.num_features)
        print('Number of Classes:', self.num_classes)
        print('Hidden Layers:')
        print('--------------')
        for i in range(1, self.L-1):
            print('Layer {}, Units {}'.format(i, self.layers[i]))
        print('--------------')
        print('Number of parameters:', num_params)

    def train_on_batch(self, X, Y, lr):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            A = self.forward_prop(X)
            loss = self.compute_loss(A, Y)

        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])

        del tape
        self.update_params(lr)
        return loss.numpy()

    def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
        history = {'val_loss': [], 'train_loss': [], 'val_acc': [], 'train_acc': []}

        for e in range(0, epochs):
            epoch_train_loss = 0.
            print('Epoch {}'.format(e), end=' ')
            for i in range(0, steps_per_epoch):
                x_batch = x_train[i * batch_size : (i + 1) * batch_size]
                y_batch = y_train[i * batch_size : (i + 1) * batch_size]

                batch_loss = self.train_on_batch(x_batch, y_batch, lr)
                epoch_train_loss += batch_loss

                if i % int(steps_per_epoch / 10) == 0:
                    print(end='=')

            train_loss = epoch_train_loss / steps_per_epoch
            history['train_loss'].append(train_loss)
            val_A = self.forward_prop(x_test)
            train_A = self.forward_prop(x_train)
            val_loss = self.compute_loss(val_A, y_test).numpy()
            history['val_loss'].append(val_loss)
            val_preds = self.predict(x_test)
            train_preds = self.predict(x_train)
            val_acc = np.sum(np.argmax(y_test, axis=1) == val_preds.numpy()) / len(y_test)
            train_acc = np.mean(np.argmax(y_train, axis=1) == train_preds.numpy())
            history['val_acc'].append(val_acc)
            history['train_acc'].append(train_acc)
            print(' \n Train loss', train_loss)
            print(' Val loss', val_loss)
            print(' Train acc', train_acc)
            print(' Val acc', val_acc)
        return history
```
## Application
### Loading Data
```
(x_train, y_train), (x_test, y_test) = nn_utils.load_data()
nn_utils.plot_random_examples(x_train, y_train).show()
```
### Network Initialization and Training
```
net = NeuralNetwork([784, 256, 128, 128, 64, 32, 10])
net.info()

batch_size = 128
epochs = 50
steps_per_epoch = int(x_train.shape[0] / batch_size)
lr = 3e-3

print('Steps per epoch', steps_per_epoch)

history = net.train(x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr)

nn_utils.plot_results(history).show()

preds = net.predict(x_test)

nn_utils.plot_random_examples(x_test, y_test, preds.numpy()).show()
```

## Utility Functions (nn_utils)

The nn_utils module should contain the necessary utility functions for loading data, plotting examples, and plotting results. Make sure to implement or import these functions accordingly.
## Conclusion

This project demonstrates the implementation of a neural network from scratch using TensorFlow's core functionality, without relying on the high-level Keras API. It covers the fundamental steps of neural network training and provides a framework for further development and experimentation.
