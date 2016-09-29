# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # tensorflow describes computations as nodes and edges in a graph
  # the nodes are computations and the edges are the inputs/outputs.
  # edges are usually/always? vectors.
  #
  # no computation happens until the session is run.
  # all computation is optimized to be computed within the tensorflow framework,
  # which can be executed on various hardware.

  # input nodes, placeholders are always input data
  x = tf.placeholder(tf.float32, [None, 784])
  # node edges between input layer and "neuron" layer
  # edges in the computation graph are called 'tensors'
  W = tf.Variable(tf.zeros([784, 10])) 
  # Variables are settable tensors within the tensorflow session
  # Variables must be initialized before than can be used in a session
  b = tf.Variable(tf.zeros([10])) # biases for neuron layer
  y = tf.matmul(x, W) + b # output of network feed-forward computation

  # actual classifications from data set
  y_ = tf.placeholder(tf.float32, [None, 10]) 

  # average cost for output nodes (y) vs actual data (y_)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) 
  # runs gradient descent with learning-rate 0.5 against cost function
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 
  
  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000): #train for 1000 epochs
    batch_xs, batch_ys = mnist.train.next_batch(100) # use mini-batch size of 100
    # feed_dict replaces placeholder values or any other tensors in your graph 
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  # argmax(y) grabs highest probability predicted class, argmax(y_) grabs actual class
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # cast booleans from prediction to floats and average results for overall accuracy 
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
