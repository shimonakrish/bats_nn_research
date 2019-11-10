# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow as tf
import numpy as np

import vgg_input as vgg_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
cpu = '/cpu:0'
gpu = '/gpu:0'

batch_size = 1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lambda1', 0.0001,
                            """lambda.""")
tf.app.flags.DEFINE_integer('batch_size', batch_size,
                            """Number of images to process in a batch.""")
data_dirs = ['C:/Users/shimon/Documents/Visual Studio 2015/Projects/3.4.19_batch/']
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")



NUM_CLASSES = vgg_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = vgg_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = vgg_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 18.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def orthogonal_initializer(shape, dtype):
    if (FLAGS.eval_data == 'test' or FLAGS.Resume):
      return tf.zeros(shape)
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape) #this needs to be corrected to float32
    print('you have initialized one orthogonal matrix.')
    return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)


def ang_diff(angle1, angle2):    
    return tf.abs(tf.minimum(tf.abs(angle1 - angle2), 360 - tf.abs(angle1 - angle2)) )  #6 - shimon

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """

  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_device(name, initializer, trainable, device, regularizer=None, shape=None):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """

  with tf.device(device):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(initializer=initializer,name = name, dtype=dtype, 
                          trainable=trainable, regularizer=regularizer,shape=shape)#, collections=collections)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, trainable, device):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  init = tf.orthogonal_initializer if trainable else tf.zeros_initializer
  if wd is not None:
    regularizer = tf.contrib.layers.l2_regularizer(wd)
  else:
    regularizer = None
  var = _variable_on_device(
      name,
      init,trainable, device,regularizer,shape)

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dirs
  """
  if not data_dirs:
    raise ValueError('Please supply a data_dirs')
  images, labels = vgg_input.distorted_inputs(data_dirs=data_dirs,
                                                  batch_size=FLAGS.batch_size)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dirs
  """
  if not data_dirs:
    raise ValueError('Please supply a data_dirs')
  images, labels = vgg_input.inputs(eval_data=eval_data,
                                        data_dirs=data_dirs,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images,device):
  """Build the CIFAR-10 model vgg16.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1_1

 # keep = tf.cond(is_training,
 #   lambda: 0.8,
 #   lambda: 1.0)

  keep = 1.0
  variabels = {}
  activations = {}
  with tf.device(device):
    parameters = []
    keep_prob = keep
    data_dropout = tf.nn.dropout(images, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')
    with tf.variable_scope('conv1_1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 3, 64],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(data_dropout, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [64])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv1_1 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv1_1)
          parameters += [kernel, biases]
  
          variabels['conv1_1/weights'] = kernel
          variabels['conv1_1/biases'] = biases
          activations['conv1_1'] = conv1_1
    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 64, 64],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [64])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv1_2 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv1_2)
          parameters += [kernel, biases]

          variabels['conv1_2/weights'] = kernel
          variabels['conv1_2/biases'] = biases
          activations['conv1_2'] = conv1_2
    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool1')
  
    data_dropout = tf.nn.dropout(pool1, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 64, 128],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(data_dropout, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [128])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv2_1 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv2_1)
          parameters += [kernel, biases]
          variabels['conv2_1/weights'] = kernel
          variabels['conv2_1/biases'] = biases
          activations['conv2_1'] = conv2_1

      # conv2_2
    with tf.variable_scope('conv2_2') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 128, 128],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [128])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv2_2 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv2_2)
          parameters += [kernel, biases]
          variabels['conv2_2/weights'] = kernel
          variabels['conv2_2/biases'] = biases
          activations['conv2_2'] = conv2_2
    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool2')
  
    data_dropout = tf.nn.dropout(pool2, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # conv3_1
    with tf.variable_scope('conv3_1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 128, 256],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(data_dropout, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [256])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv3_1 = tf.nn.relu( out, name=scope.name)
          _activation_summary(conv3_1)
          parameters += [kernel, biases]
          variabels['conv3_1/weights'] = kernel
          variabels['conv3_1/biases'] = biases
          activations['conv3_1'] = conv3_1
    # conv3_2
    with tf.variable_scope('conv3_2') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 256, 256],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [256])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv3_2 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv3_2)
          parameters += [kernel, biases]
          variabels['conv3_2/weights'] = kernel
          variabels['conv3_2/biases'] = biases
          activations['conv3_2'] = conv3_2
    # conv3_3
    with tf.variable_scope('conv3_3') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 256, 256],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [256])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv3_3 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv3_3)
          parameters += [kernel, biases]
          variabels['conv3_3/weights'] = kernel
          variabels['conv3_3/biases'] = biases
          # conv3_4
    with tf.variable_scope('conv3_4') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 256, 256],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv3_3, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [256])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv3_4 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv3_4)
          parameters += [kernel, biases]
          variabels['conv3_4/weights'] = kernel
          variabels['conv3_4/biases'] = biases
          activations['conv3_4'] = conv3_4
    # pool3
    pool3 = tf.nn.max_pool(conv3_4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool3')

    data_dropout = tf.nn.dropout(pool3, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # conv4_1
    with tf.variable_scope('conv4_1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 256, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(data_dropout, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv4_1 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv4_1)
          parameters += [kernel, biases]
          variabels['conv4_1/weights'] = kernel
          variabels['conv4_1/biases'] = biases
          activations['conv4_1'] = conv4_1
    # conv4_2
    with tf.variable_scope('conv4_2') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv4_2 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv4_2)
          parameters += [kernel, biases]
          variabels['conv4_2/weights'] = kernel
          variabels['conv4_2/biases'] = biases
          activations['conv4_2'] = conv4_2
    # conv4_3
    with tf.variable_scope('conv4_3') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv4_3 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv4_3)
          parameters += [kernel, biases]
          variabels['conv4_3/weights'] = kernel
          variabels['conv4_3/biases'] = biases
          activations['conv4_3'] = conv4_3
    # conv4_4
    with tf.variable_scope('conv4_4') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv4_3, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv4_4 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv4_4)
          parameters += [kernel, biases]
          variabels['conv4_4/weights'] = kernel
          variabels['conv4_4/biases'] = biases
          activations['conv4_4'] = conv4_4
    # pool4
    pool4 = tf.nn.max_pool(conv4_4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool4')
    data_dropout = tf.nn.dropout(pool4, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # conv5_1
    with tf.variable_scope('conv5_1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(data_dropout, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv5_1 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_1)
          parameters += [kernel, biases]
          variabels['conv5_1/weights'] = kernel
          variabels['conv5_1/biases'] = biases
          activations['conv5_1'] = conv5_1
    # conv5_2
    with tf.variable_scope('conv5_2') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv5_2 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_2)
          parameters += [kernel, biases]
          variabels['conv5_2/weights'] = kernel
          variabels['conv5_2/biases'] = biases
          activations['conv5_2'] = conv5_2
    # conv5_3
    with tf.variable_scope('conv5_3') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv5_3 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_3)
          parameters += [kernel, biases]
          variabels['conv5_3/weights'] = kernel
          variabels['conv5_3/biases'] = biases
          activations['conv5_3'] = conv5_3
    # conv5_4
    with tf.variable_scope('conv5_4') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=False, device=device)#0.0
          conv = tf.nn.conv2d(conv5_3, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=False, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out, trainable=False)
          conv5_4 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_4)
          parameters += [kernel, biases]
          variabels['conv5_4/weights'] = kernel
          variabels['conv5_4/biases'] = biases
          activations['conv5_4'] = conv5_4

   # conv5_5
    with tf.variable_scope('conv5_5') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=True, device=device)#0.0
          conv = tf.nn.conv2d(conv5_4, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out)
          conv5_5 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_5)
          parameters += [kernel, biases]
          variabels['conv5_5/weights'] = kernel
          variabels['conv5_5/biases'] = biases
          activations['conv5_5'] = conv5_5
   
    # conv5_6
    with tf.variable_scope('conv5_6') as scope:
          kernel = _variable_with_weight_decay('weights',
                                                  shape=[3, 3, 512, 512],                                                     
                                                  stddev=1e-2,
                                                  wd=0.0005, trainable=True, device=device)#0.0
          conv = tf.nn.conv2d(conv5_5, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [512])
          out = tf.nn.bias_add(conv, biases)
          out = tf.layers.batch_normalization(out)
          conv5_6 = tf.nn.relu(out, name=scope.name)
          _activation_summary(conv5_6)
          parameters += [kernel, biases]
          variabels['conv5_6/weights'] = kernel
          variabels['conv5_6/biases'] = biases
          activations['conv5_6'] = conv5_6

     #pool5
    pool5 = tf.nn.max_pool(conv5_6,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool5')
    data_dropout = tf.nn.dropout(pool5, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # fc1
    with tf.variable_scope('fc1') as scope:         
          reshape = tf.reshape(data_dropout, [FLAGS.batch_size, -1])
          dim = reshape.get_shape()[1].value
          weights = _variable_with_weight_decay('weights', shape=[dim, 4096],
                                                  stddev=1e-2, wd=0.0005, trainable=True, device=device)#0.004
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [4096])#0.1
          fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
          _activation_summary(fc1)
          variabels['fc1/weights'] = weights
          variabels['fc1/biases'] = biases
          activations['fc1'] = fc1
    data_dropout = tf.nn.dropout(fc1, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')

    # fc2
    with tf.variable_scope('fc2') as scope:
          weights = _variable_with_weight_decay('weights', shape=[4096, 1024],
                                                  stddev=1/4096.0, wd=0.0005, trainable=True, device=device)#0.004
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [1024])#0.1
          fc2 = tf.nn.relu(tf.matmul(data_dropout, weights) + biases, name=scope.name)
          _activation_summary(fc2)
          variabels['fc2/weights'] = weights
          variabels['fc2/biases'] = biases
          activations['fc2'] = fc2
    data_dropout = tf.nn.dropout(fc2, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')
    # fc3
    with tf.variable_scope('fc3') as scope:
          weights = _variable_with_weight_decay('weights', shape=[1024, 256],
                                                  stddev=1/1024.0, wd=0.0005, trainable=True, device=device)#0.004
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [256])#0.1
          fc3 = tf.nn.relu(tf.matmul(data_dropout, weights) + biases, name=scope.name)
          _activation_summary(fc3)
          variabels['fc3/weights'] = weights
          variabels['fc3/biases'] = biases
          activations['fc3'] = fc3
    data_dropout = tf.nn.dropout(fc3, rate = 1 - keep_prob, noise_shape=None, seed=None, name='data_dropout')
    with tf.variable_scope('softmax_linear') as scope:
          weights = _variable_with_weight_decay('weights', [256, 1],
                                              stddev=1/256.0, wd=5e-4, trainable=True, device=device)#0.0
          biases = _variable_on_device('biases', tf.zeros_initializer, trainable=True, device=device, shape = [1])

          softmax_linear = tf.nn.bias_add(tf.matmul(data_dropout, weights), biases)
        

          _activation_summary(softmax_linear)
          variabels['softmax_linear/weights'] = weights
          variabels['softmax_linear/biases'] = biases       

  return {'s': softmax_linear, 'p' : parameters, 'v': variabels, 'a' : activations}

  
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
 
  labels = tf.cast(labels, tf.float32)
 
  #L2
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  numer =  tf.cast((2 * batch_size), tf.float32)
  
  logits_r = tf.reshape(logits,[batch_size])

  diff = ang_diff(logits_r,labels)

  loss = (tf.reduce_sum(tf.pow(diff, 2)) / numer) + FLAGS.lambda1 * tf.reduce_sum(regularization_losses)

  
  tf.add_to_collection('losses', loss)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

  

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')#0.9
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +'_raw', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  #Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=False)


  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):

    opt = tf.train.AdagradOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


