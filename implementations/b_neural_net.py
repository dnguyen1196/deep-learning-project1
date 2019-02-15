"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with tensorflow instead of numpy
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """

    # store parameters in numpy arrays
    self.params = {}
    self.params['W1'] = tf.Variable(std * np.random.randn(input_size, hidden_size), dtype=tf.float32)
    self.params['b1'] = tf.Variable(np.zeros(hidden_size), dtype=tf.float32)
    self.params['W2'] = tf.Variable(std * np.random.randn(hidden_size, output_size), dtype=tf.float32)
    self.params['b2'] = tf.Variable(np.zeros(output_size), dtype=tf.float32)

  def softmax_loss_helper(self, scores, y):
    """
    Compute the softmax loss. Implement this function in tensorflow

    Inputs:
    - scores: Input data of shape (N, C), tf tensor. Each scores[i] is a vector 
              containing the scores of instance i for C classes .
    - y: Vector of training labels, tf tensor. y[i] is the label for X[i], and each y[i] is
         an integer in the range 0 <= y[i] < C. This parameter is optional; if it
         is not passed then we only return scores, and if it is passed then we
         instead return the loss and gradients.
    - reg: Regularization strength, scalar.

    Returns:
    - loss: softmax loss for this batch of training samples.
    """
    N, C = scores.shape
    ys = tf.one_hot(y, C)

    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=scores)
    #############################################################################
    # TODO: compute the softmax loss. please check the documentation of         # 
    # tf.nn.softmax_cross_entropy_with_logits                                   #
    #############################################################################

    return tf.reduce_mean(softmax_loss)


  def softmax_loss(self, X, y, reg=0.05):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    N, D = X.shape
    _, C = W2.get_shape()

    scores = self.compute_scores(X, C)
    ys = tf.one_hot(y, C)

    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=scores)
    softmax_loss = tf.reduce_mean(softmax_loss)

    softmax_loss += reg * (tf.reduce_sum(W1**2) + tf.reduce_sum(W2**2) + tf.reduce_sum(b1**2) + tf.reduce_sum(b2**2))

    return softmax_loss


  def compute_scores(self, X, C):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. Implement this function in tensorflow

    Inputs:
    - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.
    - C: integer, the number of classes 

    Returns:
    - scores: a tensor of shape (N, C) where scores[i, c] is the score for 
              class c on input X[i].

    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    h1 = tf.matmul(X, W1) + b1
    h1 = tf.nn.relu(h1)

    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be a tensor  of      #
    # shape (N, C).                                                             #
    #############################################################################
    scores = tf.matmul(h1, W2) + b2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    return scores


  def compute_objective(self, X, y, reg):
    """
    Compute the training objective of the neural network.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - reg: a np.float32 scalar


    Returns: 
    - objective: a tensorflow scalar. the training objective, which is the sum of 
                 losses and the regularization term
    """
    #############################################################################
    # TODO: use the function compute_scores() and softmax_loss(), also implement# 
    # the regularization term here, to compute the training objective           #
    #############################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    N, D = X.shape
    _, C = W2.get_shape()

    scores = self.compute_scores(X, C)
    ys = tf.one_hot(y, C)

    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=scores)
    softmax_loss = tf.reduce_mean(softmax_loss)

    softmax_loss += reg * (tf.reduce_sum(W1**2) + tf.reduce_sum(W2**2) + tf.reduce_sum(b1**2) + tf.reduce_sum(b2**2))

    return softmax_loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################


  def loss(self, X, y, reg):
    """
    Compute the training objective of the neural network.
    As well as the numerical gradients

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - reg: a np.float32 scalar


    Returns: 
    - loss: a tensorflow scalar. the training loss, which is the sum of 
                 losses and the regularization term
    - grads: a dictionary type with the same keys as self.params but
    the gradient is computed by hand
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    loss = self.compute_objective(X, y, reg)

    # grads = {"W1" : 0, "W2" : 0, "b1": 0, "b2":0}
    grads = {}

    grads["W1"] = tf.gradients(self.compute_objective(X, y, reg), [W1])
    grads["W2"] = tf.gradients(self.compute_objective(X, y, reg), [W2])
    grads["b1"] = tf.gradients(self.compute_objective(X, y, reg), [b1])
    grads["b2"] = tf.gradients(self.compute_objective(X, y, reg), [b2])

    return loss, grads


  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=np.float32(5e-6), num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train, D = X.shape

    iterations_per_epoch = max(num_train / batch_size, 1)

    # Get place holder for X_data and y_labels
    x_batch_placeholder = tf.placeholder(shape=[None,D], dtype=tf.float32, name='feature')
    y_batch_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name='label')

    # Define the loss function
    loss_function = self.compute_objective(X=x_batch_placeholder, y=y_batch_placeholder, reg=reg)

    # Define optimization algorithm, SGD with decaying step size
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    lr = tf.train.exponential_decay(learning_rate, global_step,
                                      decay_steps=100000, decay_rate=learning_rate_decay, staircase=True)

    # Passing global_step to minimize() will increment it at each step.
    opt = tf.train.GradientDescentOptimizer(lr) #.minimize(loss_function, global_step=global_step)

    # Define the automatic differentiation step
    grads_vars = opt.compute_gradients(loss_function, var_list=self.params)

    # Define the update step
    update = opt.apply_gradients(grads_vars, global_step=global_step)

    # by this line, you should have constructed the tensorflow graph  
    # no more graph construction
    ############################################################################
    # after this line, you should execute appropriate operations in the graph to train the mode  

    # Initialize the tensorflow graph session
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    self.session = session

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []


    # counter for cyclic batching
    counter = 0

    for it in range(num_iters):
      #########################################################################
      # Create a random minibatch of training data and labels, storing        #
      # them in X_batch and y_batch respectively.                             #
      # 
      #########################################################################
      end_counter = min(counter + batch_size, num_train)

      batch_idx = np.arange(counter, end_counter) # Cycling step size
      counter   = end_counter % num_train
      # batch_idx = np.random.choice(num_train, size=(batch_size,))
      X_batch   = np.take(X, batch_idx, axis=0)
      y_batch   = np.take(y, batch_idx, axis=0)

      # Compute loss and gradients using the current minibatch
      feed_dict = {x_batch_placeholder : X_batch, y_batch_placeholder : y_batch}
      loss_value = session.run(loss_function, feed_dict=feed_dict)

      loss_history.append(loss_value) # need to feed in the data batch

      # run the update operation to perform one gradient descending step
      session.run(update, feed_dict=feed_dict)

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 10 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss_value))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch).eval(session=session) == y_batch).mean()
        val_acc = (self.predict(X_val).eval(session=session) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
      'objective_history' : loss_history,
    }


  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    ###########################################################################
    # TODO: Implement this function.                                          #
    ###########################################################################
    # feed in X and  compute scores 
    C = self.params["b2"].shape
    y_pred = self.compute_scores(X, C)

    # take the index of the largest value in each row.
    # check tf.argmax
    y_pred = tf.argmax(y_pred, axis=1)

    # run it to get a numpy array
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

  def get_learned_parameters(self):
    parameters = {}
    for param_name in self.params:
      parameters[param_name] = self.params[param_name].eval(self.session)
    return parameters


