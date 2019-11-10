from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import operator
import tf_cnnvis
import vgg as vgg
cpu = '/cpu:0'
gpu = '/device:GPU:0'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'D:/tensorflow_fs/vgg_19_train_test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('Resume', False,
                            """Resume training from check point.""")
tf.app.flags.DEFINE_integer('num_examples', 33700,
                            """Number of examples to run.""")

tf.app.flags.DEFINE_string('eval_dir', 'D:/tensorflow_fs/vgg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:/tensorflow_fs/vgg_19_train_test',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_data', 'train',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('test_batch', 'data_batch_47_1st.bin',  """test batch""")

		   
def load_weights(parameters, weight_file, sess):
    weights = np.load(weight_file, encoding='latin1').item()
    keys = sorted(weights.items(), key=operator.itemgetter(0))
    for i, k in enumerate(keys):
      if('fc' not in k[0]):
        print(2*i, k[0], np.shape(k[1][0]), parameters[2*i].shape)
        sess.run(parameters[2*i].assign(k[1][0]))
        print(2*i+1, k[0], np.shape(k[1][1]), parameters[2*i+1].shape)
        sess.run(parameters[2*i+1].assign(k[1][1]))

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

    images_train, labels_train = vgg.distorted_inputs()

    images_val, labels_val = vgg.inputs(eval_data='test')

    is_training = tf.placeholder('bool', [], name='is_training')

    images, labels = tf.cond(is_training,
        lambda: (images_train, labels_train),
        lambda: (images_val, labels_val))


    # Build a Graph that computes the logits predictions from the
    # inference model.
    graph =  vgg.inference(images,gpu,is_training)
    logits = graph['s']
    params = graph['p']
    logits = tf.transpose(logits)
    # Calculate loss.
    loss = vgg.loss(logits, labels)


    logits_r = tf.reshape(logits,[vgg.batch_size])

    diff = vgg.ang_diff(logits_r,labels)

    true_count = tf.reduce_sum(tf.cast(tf.less(diff,25),tf.uint8))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = vgg.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    
    summary = tf.Summary()
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.

    
    config = tf.ConfigProto(allow_soft_placement = True)
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.eval_dir, sess.graph)

    if (FLAGS.Resume):
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        
        sub_saver =  tf.train.Saver(graph['v'])
        sub_saver.save(sess, FLAGS.train_dir)
      else:
        print('No checkpoint file found')
        return
    else:
      sess.run(init,{ is_training: False })
      load_weights(params,'vgg19.npy', sess)

    test_iters = 11
    total_sample_count = test_iters * FLAGS.batch_size 
    
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss],{ is_training: True })
      duration = time.time() - start_time
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step > 1 and step % 250 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
        summary_str = sess.run(summary_op,{ is_training: False })
        summary_writer.add_summary(summary_str, step)
        
        summary.ParseFromString(sess.run(summary_op,{ is_training: False }))
        summary_writer.add_summary(summary, step)


      
      if step > 1 and step % 1000== 0 or (step + 1) == FLAGS.max_steps:     

        true_count_sum = 0  # Counts the number of correct predictions.
        diffs = np.array([])
        for i in range(test_iters): 
          true_count_ ,diff_= sess.run([true_count,tf.unstack(diff)],{ is_training: False })
          true_count_sum += true_count_
          diffs = np.append(diffs,diff_)
          
        diffs_var  = np.var(diffs)
        diffs_mean = np.mean(diffs)
        
        # Compute precision @ 1.
        precision = true_count_sum / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        summary.ParseFromString(sess.run(summary_op,{ is_training: False }))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary.value.add(tag='diffs_var', simple_value=diffs_var)
        summary.value.add(tag='diffs_mean', simple_value=diffs_mean)  
        summary_writer.add_summary(summary, step)
        # Save the model checkpoint periodically.
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
