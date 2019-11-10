from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
from os import listdir
from os.path import isfile, join
import tensorflow as tf


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

RSZ_H = 0
RSZ_W = 0

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 33700
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3238
NUM_CLASSES = 1
FLAGS = tf.app.flags.FLAGS


def read_vgg(filename_queue):
  """Reads and parses examples from vgg data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class vggRecord(object):
    pass
  result = vggRecord()

  label_bytes = 4  
  result.height = IMAGE_HEIGHT
  result.width = IMAGE_WIDTH
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  channel_bytes = result.height * result.width

  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
 # reader = tf.data.FixedLengthRecordDataset(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  record_bytes2 = tf.decode_raw(value, tf.float32)
  #tf.Print(record_bytes,[record_bytes],message=None, first_n=None, summarize=None, name=None)
  # The first bytes represent the label, which we convert from uint8->int32.


  result.label = tf.slice(record_bytes2, [0], [1])
  
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  R = tf.reshape(tf.slice(record_bytes, [label_bytes], [channel_bytes]),
                           [result.height, result.width])
  G = tf.reshape(tf.slice(record_bytes, [label_bytes + channel_bytes], [channel_bytes]),
                           [result.height, result.width])
  B = tf.reshape(tf.slice(record_bytes, [label_bytes + (2 * channel_bytes)], [channel_bytes]),
                           [result.height, result.width])

  G = tf.zeros([160,160],tf.uint8)

  depth_major = tf.stack([R,G,B])
  
  #depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
  #                         [result.depth, result.height, result.width])


  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  #result.uint8image = depth_major

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #num_preprocess_threads = 16
  num_preprocess_threads = 1
  if shuffle:
  #  images, label_batch = tf.data.Dataset.shuffle(min_after_dequeue=min_queue_examples).batch(batch_size)
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  
  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dirs, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dirs: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  for dir in data_dirs:
    filenames = [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and FLAGS.test_batch not in f)]

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  #filename_queue = tf.data.Dataset.from_tensor_slices(filenames)
  # Read examples from files in the filename queue.
  read_input = read_vgg(filename_queue)
  float_image = tf.cast(read_input.uint8image, tf.float32)


  resize_factor = 1 + tf.random.uniform([1])
  resized_image = tf.image.resize_images(float_image, [height*resize_factor,width*resize_factor], align_corners=False) # shimon
  # Crop the central [height, width] of the image.
  image_crop = tf.image.resize_image_with_crop_or_pad(resized_image,[height,width])
  # Subtract off the mean and divide by the variance of the pixels.
  #image = tf.image.per_image_standardization(resized_image)
  #image=resized_image
  image = image_crop



  # Image processing for evaluation.
    
  #croped_image = tf.random_crop(standrd_image, [height - RSZ_H, width - RSZ_W, 3])
  # Randomly flip the image horizontally.
  #image = tf.image.random_flip_left_right(standrd_image)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  
  # Set the shapes of tensors.
  image.set_shape([height - RSZ_H, width - RSZ_W, 3])
  read_input.label.set_shape([1])

  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dirs, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dirs: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  if eval_data == 'test':
    for dir in data_dirs:
      filenames = [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and FLAGS.test_batch in f)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  else:
    for dir in data_dirs:
      filenames = [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and FLAGS.test_batch not in f)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_vgg(filename_queue)
  float_image = tf.cast(read_input.uint8image, tf.float32)


 
  image = float_image
  # Set the shapes of tensors.
  image.set_shape([height - RSZ_H, width - RSZ_W, 3])


  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
