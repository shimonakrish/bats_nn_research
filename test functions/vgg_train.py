from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import operator
import vgg as vgg
import pylab
import imageio
import geopy.distance as dist
import geopy as gp
import csv
import simplekml
import cv2
import tf_cnnvis as vis
import tf_cnnvis_verbose as vis2
import random
import scipy.io

cpu = '/cpu:0'
gpu = '/device:GPU:0'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_data', 'train',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('train_dir', 'D:/tensorflow_fs/vgg19_train_holes/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('Resume', False,
                            """Resume training from check point.""")


tf.app.flags.DEFINE_string('eval_dir', 'D:/tensorflow_fs/vgg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:/tensorflow_fs/vgg19_train_holes/',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_float('bearing_bias', 0, """bearing bias""")
tf.app.flags.DEFINE_string('test_batch', 'data_batch_010.bin',  """test batch""")
tf.app.flags.DEFINE_integer('test_batch_size', 995, """test batch""")

def HeadingTo (src,dest):

    φ1 = np.deg2rad(src[0])#src lat
    φ2 = np.deg2rad(dest[0])#dest lat
    Δλ = np.deg2rad(dest[1]-src[1]) #dest lon - src lon
    # if dLon over 180° take shorter rhumb line across the anti-meridian:
    if (Δλ >  np.pi):
      Δλ -= 2 * np.pi
    if (Δλ < -np.pi):
      Δλ += 2 * np.pi

    Δψ = np.log(np.tan(φ2/2+np.pi/4)/np.tan(φ1/2+np.pi/4))

    θ = np.arctan2(Δλ, Δψ)

    return (np.rad2deg(θ)+360) % 360;

#---------------------------------------------------------------------------------------------------------
def destinationPoint(lat,lon, distance, bearing):
    radius = 6371e3

    δ = distance / radius #// angular distance in radians
    θ = np.deg2rad(bearing)
    

    φ1 = np.deg2rad(lat)
    λ1 = np.deg2rad(lon)

    sinφ1 = np.sin(φ1)
    cosφ1 = np.cos(φ1)
    sinδ = np.sin(δ) 
    cosδ = np.cos(δ)
    sinθ = np.sin(θ) 
    cosθ = np.cos(θ)

    sinφ2 = sinφ1*cosδ + cosφ1*sinδ*cosθ
    φ2 = np.arcsin(sinφ2)
    y = sinθ * sinδ * cosφ1
    x = cosδ - sinφ1 * sinφ2
    λ2 = λ1 + np.arctan2(y, x)

    return np.rad2deg(φ2), (np.rad2deg(λ2)+540)%360-180 #// normalise to −180..+180°
#---------------------------------------------------------------------------------------------------------
def find_closest_points(lat,lon, heading,size):  
  gps_labels = open("E:/Final Project/walk/gps_labels/gps_labels.csv")
  reader = csv.reader(gps_labels,delimiter =",")
  ang_diff_limit = [*range(-size,-2,2),*range(4,size+2,2)]
  length = len(ang_diff_limit)
  cord_list = (length-1) * [[]]
  min_d = (length-1) * [1e10]
  for row in reader:     
    db_heading = float(row[5])
    db_lat = float(row[0])
    db_lon = float(row[1])
    valid = int(row[6])
    d = dist.distance((lat,lon),(db_lat, db_lon)).m
    ang_diff_abs = np.minimum(np.abs(db_heading - heading), 360 - np.abs(db_heading - heading))
    db_h_shift = (db_heading + (360 - heading))%360
    ang_diff_sign = np.sign(db_h_shift - 180)
    ang_diff = ang_diff_abs * ang_diff_sign

    for i in range(length-1):
      if (d < min_d[i] and ang_diff > ang_diff_limit[i] and ang_diff < ang_diff_limit[i+1] and valid):
        min_d[i] = d
        cord_list[i] = row,d,ang_diff


  return cord_list 
#---------------------------------------------------------------------------------------------------------
def find_closest_points2(lat,lon, heading):  
  gps_labels = open("E:/Final Project/walk/gps_labels/gps_labels.csv")
  reader = csv.reader(gps_labels,delimiter =",")
  ang_diff_limit = [0,4]
  length = len(ang_diff_limit)
  cord_list = (length-1) * [[]]
  min_d = (length-1) * [1e10]
  for row in reader:     
    db_heading = float(row[5])
    db_lat = float(row[0])
    db_lon = float(row[1])
    valid = int(row[6])
    d = dist.distance((lat,lon),(db_lat, db_lon)).m
    ang_diff_abs = np.minimum(np.abs(db_heading - heading), 360 - np.abs(db_heading - heading))
    db_h_shift = (db_heading + (360 - heading))%360
    ang_diff_sign = np.sign(db_h_shift - 180)
    ang_diff = ang_diff_abs * ang_diff_sign

    for i in range(length-1):
      if (d < min_d[i] and ang_diff_abs > ang_diff_limit[i] and ang_diff_abs < ang_diff_limit[i+1] and valid):
        min_d[i] = d
        cord_list[i] = row,d,ang_diff

  return cord_list 
#---------------------------------------------------------------------------------------------------------
def bearing_compensation(db_cord,curr_cord):
  dest = (31.669251,34.741851)
  db_hd_2_dest = HeadingTo(db_cord,dest)
  curr_hd_2_dest = HeadingTo(curr_cord,dest)
  return (curr_hd_2_dest - db_hd_2_dest)
#---------------------------------------------------------------------------------------------------------
def load_weights(parameters, weight_file, sess):
    weights = np.load(weight_file, encoding='latin1').item()
    keys = sorted(weights.items(), key=operator.itemgetter(0))
    for i, k in enumerate(keys):
      if('fc' not in k[0]):
        print(2*i, k[0], np.shape(k[1][0]), parameters[2*i].shape)
        sess.run(parameters[2*i].assign(k[1][0]))
        print(2*i+1, k[0], np.shape(k[1][1]), parameters[2*i+1].shape)
        sess.run(parameters[2*i+1].assign(k[1][1]))
#---------------------------------------------------------------------------------------------------------
def test_vid():
  video = cv2.VideoWriter(filename='video.avi',fourcc=cv2.VideoWriter_fourcc('X','V','I','D'),fps=24,frameSize=(640,640))
  filename = 'E:/Final Project/walk/vids/DJI_0033.mp4'
  vid = imageio.get_reader(filename,'ffmpeg')
  for i in range(1000):
    img = vid.get_data(i)
    img_rsz = imresize(img,[640,640,3])
    video.write(img_rsz)
  cv2.destroyAllWindows()
  video.release()
#---------------------------------------------------------------------------------------------------------
def cnnvis(fd,sess):
  layers = ['c']
  with sess.as_default():
    vis2.activation_visualization(sess_graph_path = None, value_feed_dict = fd, layers=layers, path_logdir='./cnnvis/Log_act', path_outdir='./cnnvis/Output_act')



#---------------------------------------------------------------------------------------------------------
def image_test():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    img_name = '5_2.BMP'
    img_normal = imread('E:/final Project/3.4.19/croped vids/matlab normal/' + img_name)
    img_top = imread('E:/final Project/3.4.19/croped vids/matlab top/' + img_name)
    img_middle = imread('E:/final Project/3.4.19/croped vids/matlab middle/' + img_name)
    img_bottom = imread('E:/final Project/3.4.19/croped vids/matlab bottom/' + img_name)
    #img_rsz = imresize(img,[160,160,3])

    img_normal = np.transpose(img_normal, [1, 0, 2])
    img_normal_in = np.reshape(img_normal,[1,160,160,3])

    img_top = np.transpose(img_top, [1, 0, 2])
    img_top_in = np.reshape(img_top,[1,160,160,3])

    img_middle = np.transpose(img_middle, [1, 0, 2])
    img_middle_in = np.reshape(img_middle,[1,160,160,3])

    img_bottom = np.transpose(img_bottom, [1, 0, 2])
    img_bottom_in = np.reshape(img_bottom,[1,160,160,3])

    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))
    
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    logits = tf.transpose(logits)
    logits_r = tf.reshape(logits,[vgg.batch_size])
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)  
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt)

    out_normal = sess.run(logits, feed_dict={image: img_normal_in})[0][0] 
    out_top = sess.run(logits, feed_dict={image: img_top_in})[0][0] 
    out_middle = sess.run(logits, feed_dict={image: img_middle_in})[0][0] 
    out_bottom = sess.run(logits, feed_dict={image: img_bottom_in})[0][0] 

    print(img_name)
    print('normal: ',out_normal)
    print('top: ',out_top)
    print('middle: ',out_middle)
    print('bottom: ',out_bottom)
    print('done')
#---------------------------------------------------------------------------------------------------------
def image_list_test():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))
    
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    logits = tf.transpose(logits)
    logits_r = tf.reshape(logits,[vgg.batch_size])
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)  
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt)
    image_list = os.listdir('E:/final Project/3.4.19/croped vids/matlab normal/')
    normal_result = []
    top_result = []
    middle_result = []
    bottom_result = []
    for img_name in image_list:

      img_normal = imread('E:/final Project/3.4.19/croped vids/matlab normal/' + img_name)
      img_top = imread('E:/final Project/3.4.19/croped vids/matlab top/' + img_name)
      img_middle = imread('E:/final Project/3.4.19/croped vids/matlab middle/' + img_name)
      img_bottom = imread('E:/final Project/3.4.19/croped vids/matlab bottom/' + img_name)


      img_normal = np.transpose(img_normal, [1, 0, 2])
      img_normal_in = np.reshape(img_normal,[1,160,160,3])

      img_top = np.transpose(img_top, [1, 0, 2])
      img_top_in = np.reshape(img_top,[1,160,160,3])

      img_middle = np.transpose(img_middle, [1, 0, 2])
      img_middle_in = np.reshape(img_middle,[1,160,160,3])

      img_bottom = np.transpose(img_bottom, [1, 0, 2])
      img_bottom_in = np.reshape(img_bottom,[1,160,160,3])

      out_normal = sess.run(logits, feed_dict={image: img_normal_in})[0][0] 
      out_top = sess.run(logits, feed_dict={image: img_top_in})[0][0] 
      out_middle = sess.run(logits, feed_dict={image: img_middle_in})[0][0] 
      out_bottom = sess.run(logits, feed_dict={image: img_bottom_in})[0][0] 

      normal_result.append(out_normal)
      top_result.append(out_top)
      middle_result.append(out_middle)
      bottom_result.append(out_bottom)
    
    print('done')
#---------------------------------------------------------------------------------------------------------
def image_test_custom():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    img_name = '9_363_3.BMP'
    img_normal = imread('E:/final Project/3.4.19/croped vids/matlab custom/' + img_name)

    img_normal = np.transpose(img_normal, [1, 0, 2])
    img_normal_in = np.reshape(img_normal,[1,160,160,3])

    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))
    
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    logits = tf.transpose(logits)
    logits_r = tf.reshape(logits,[vgg.batch_size])
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)  
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt)

    out_normal = sess.run(logits, feed_dict={image: img_normal_in})[0][0] 



    print('custom: ',out_normal)

    print('done')

#---------------------------------------------------------------------------------------------------------
def batch_test():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    images, labels = vgg.inputs(eval_data='test')
    graph = vgg.inference(images,gpu)
    logits = graph['s']
    logits = tf.transpose(logits)
    # Calculate loss.
    logits_r = tf.reshape(logits,[vgg.batch_size])

    diff = vgg.ang_diff(logits_r,labels)

    true_count = tf.reduce_sum(tf.cast(tf.less(diff,25),tf.uint8))
    saver = tf.train.Saver(tf.global_variables())
       
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg_19_train_cont/')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt)

    true_count_sum = 0  # Counts the number of correct predictions.
    diffs = np.array([])
    for i in range(FLAGS.test_batch_size): 
      true_count_ ,diff_= sess.run([true_count,tf.unstack(diff)])
      true_count_sum += true_count_
      diffs = np.append(diffs,diff_)
      if diff_[0] <= 25:
        print(i," :",diff_[0])
          
    diffs_var  = np.var(diffs)
    diffs_mean = np.mean(diffs)
        
    # Compute precision @ 1.
    precision = true_count_sum / (FLAGS.test_batch_size*FLAGS.batch_size)
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    print('done')

#--------------------------------------------------------------------------------------------------------
def cnn_vis():
  with tf.Graph().as_default():
    img_name = '9_363.BMP'
    img_normal = imread('E:/final Project/3.4.19/croped vids/matlab normal/' + img_name)
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))      
    img_normal = np.transpose(img_normal, [1, 0, 2])
    img_normal_in = np.reshape(img_normal,[1,160,160,3])
    
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    logits = tf.transpose(logits)
    logits_r = tf.reshape(logits,[vgg.batch_size])
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)  
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt)
    cnnvis({image: img_normal_in},sess)

    print('done')
#---------------------------------------------------------------------------------------------------------
def walk():
  date = datetime.now()
  date_str = date.strftime("%d")+date.strftime("%m")+date.strftime("%y")+'_'+date.strftime("%H")+date.strftime("%M")+date.strftime("%S")
  kml_name = date_str + '.kml'
  _step_dist = 350
  size = 5 
  lat = 31.62489921
  lon = 34.84234767
  bearing = 0.5754
  heading = 296.9
  
  kml=simplekml.Kml()
  kml.newpoint(name='0', coords=[(lon,lat)])
  kml.save(kml_name)


  with tf.Graph().as_default() as g:
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))
          
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)   
    init = tf.global_variables_initializer()
    sess.run(init)

    summary = tf.Summary()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('D:/tensorflow_fs/vgg_eval/', g)
    saver = tf.train.Saver(graph['v'])
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg_19_train_test/RG_0_622/')
    saver.restore(sess, ckpt)

    point_list = []
    cord_list = []
    ang_diff_list = []
    compare_list = []

    step = 0
    while ((dist.distance((31.669251,34.741851),(lat,lon)).m > 150) and step < 40):

      next_heading = (heading + bearing)%360


      step_dist = _step_dist

      new_lat,new_lon = destinationPoint(lat, lon, step_dist, next_heading)
      points = find_closest_points(new_lat, new_lon, next_heading,size)
      
      
      cord_list += [[new_lat, new_lon, bearing, next_heading, heading]]
      
      heading = HeadingTo([lat, lon], [new_lat, new_lon])

        
      lat, lon = new_lat, new_lon

      if (points[0] == []):
        raise 'did not find point'
      point_list += [points]
      ang_diff_list +=[[points[0][1],points[0][2]]]

      kml.newpoint(name=str(step+1), coords=[(new_lon,new_lat)])
      kml.save(kml_name)

      bearings = [[]]*len(points)
      for p in range(len(points)):
        if (points[p] != []):
          ang_diff = points[p][2]
          file_name = points[p][0][3]
          frame_num = int(points[p][0][4])
          p_lat = float(points[p][0][0])
          p_lon = float(points[p][0][1])
          if (file_name in ['33','34','35', '46', '47', '50', '51_1', '53', '54']):
            filetype = '.mp4'
          else:
            filetype = '.mov'

          filename = 'E:/Final Project/walk/vids/DJI_00' + file_name + filetype   
          vid = imageio.get_reader(filename,'ffmpeg')

          img = vid.get_data(frame_num)
          img_rsz_t = imresize(img,[160,160,3])


          if (p == 0):
            print('vid: '+ str(file_name) + ', frame num: '+ str(frame_num) + ', filetype: ' + filetype)


          img_in = np.reshape(img_rsz_t,[1,160,160,3])

          b_c = bearing_compensation([p_lat,p_lon],[lat,lon])

          bearings[p] = (sess.run(logits, feed_dict={image: img_in,is_training: False})[0][0] - ang_diff + b_c)


      bearing_avg = np.average(bearings)
      compare_list += [bearings,bearing_avg,[z[2] for z in points]]
      bearing = bearing_avg
      step += 1


  cv2.destroyAllWindows()

  print(cord_list)
  #---------------------------------------------------------------------------------------------------------
def batch_walk():
  with tf.Graph().as_default() as g:
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))       
    graph = vgg.inference(image, gpu)
    logits = graph['s']
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)   
    init = tf.global_variables_initializer()
    sess.run(init)
    summary = tf.Summary()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('D:/tensorflow_fs/vgg_eval/', g)
    saver = tf.train.Saver(graph['v'])
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    saver.restore(sess, ckpt)
    for run in range(50):
      date = datetime.now()
      date_str = date.strftime("%d")+date.strftime("%m")+date.strftime("%y")+'_'+date.strftime("%H")+date.strftime("%M")+date.strftime("%S")
      kml_name = date_str +'run_' + str(run) + '.kml'
      f = open("run_log_" + str(run) + '.txt', "a")
      _step_dist = 350
      size = 5 
      init_step_dist = random.randint(1000,5000)
      lat, lon = destinationPoint(31.62489921, 34.84234767, init_step_dist, 296.9) 
      bearing = 0
      heading = 296.9 + (random.randint(0,30) - 60)
      kml=simplekml.Kml()
      kml.newpoint(name='0', coords=[(lon,lat)])
      kml.save(kml_name)
      step = 0
      while ((dist.distance((31.669251,34.741851),(lat,lon)).m > 150) and step < 40):
        next_heading = (heading + bearing)%360
        step_dist = _step_dist
        new_lat,new_lon = destinationPoint(lat, lon, step_dist, next_heading)
        points = find_closest_points(new_lat, new_lon, next_heading,size)    
        heading = HeadingTo([lat, lon], [new_lat, new_lon])       
        lat, lon = new_lat, new_lon
        f.write(str(step) + ' :' + str(lat) + ',' + str(lon)+ ',' + str(heading) + '\n')
        if (points[0] == []):
          raise 'did not find point'
        kml.newpoint(name=str(step+1), coords=[(new_lon,new_lat)])
        kml.save(kml_name)
        bearings = [[]]*len(points)
        for p in range(len(points)):
          if (points[p] != []):
            ang_diff = points[p][2]
            file_name = points[p][0][3]
            frame_num = int(points[p][0][4])
            p_lat = float(points[p][0][0])
            p_lon = float(points[p][0][1])
            if (file_name in ['33','34','35', '46', '47', '50', '51_1', '53', '54']):
              filetype = '.mp4'
            else:
              filetype = '.mov'

            filename = 'E:/Final Project/walk/vids/DJI_00' + file_name + filetype   
            vid = imageio.get_reader(filename,'ffmpeg')

            img = vid.get_data(frame_num)
            img_rsz = imresize(img,[160,160,3])
            img_rsz[:,:,1] = 0
            img_in = np.reshape(img_rsz,[1,160,160,3])
            b_c = bearing_compensation([p_lat,p_lon],[lat,lon])
            bearings[p] = (sess.run(logits, feed_dict={image: img_in})[0][0] - ang_diff + b_c)
            if (p == 0):
              print('vid: '+ str(file_name) + ', frame num: '+ str(frame_num) + ', filetype: ' + filetype)

        bearing_avg = np.average(bearings)
        bearing = bearing_avg
        step += 1
      f.close()
  

#---------------------------------------------------------------------------------------------------------
def location_test():
  size = 8 
  lat =    31.653862
  lon =    34.768527
  heading = 10
  heading_2_dest = HeadingTo ([lat, lon],[31.669243,34.742072])
  with tf.Graph().as_default() as g:
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))
          
    graph = vgg.inference(image, gpu, tf.constant(False))
    logits = graph['s']
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)   
    init = tf.global_variables_initializer()
    sess.run(init,{is_training: False})
    saver = tf.train.Saver(graph['v'])
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    saver.restore(sess, ckpt)

    points = find_closest_points(lat, lon,heading,size)
    if (points[0] == []):
      raise 'did not find point'

    bearings = [[]]*len(points)
    locations = [[]]*len(points)
    for p in range(len(points)):
      if (points[p] != []):
        ang_diff = points[p][2]
        file_name = points[p][0][3]
        frame_num = int(points[p][0][4])
        p_lat = float(points[p][0][0])
        p_lon = float(points[p][0][1])
        locations[p] = [p_lat,p_lon]
        if (file_name in ['33','34','35', '46', '47', '50', '51_1', '53', '54']):
          filetype = '.mp4'
        else:
          filetype = '.mov'

        filename = 'E:/Final Project/walk/vids/DJI_00' + file_name + filetype   
        vid = imageio.get_reader(filename,'ffmpeg')

        img = vid.get_data(frame_num)
        img_rsz = imresize(img,[160,160,3])
        img_rsz[:,:,1] = 0
        if (p == 2):
          print('vid: '+ str(file_name) + ', frame num: '+ str(frame_num) + ', filetype: ' + filetype)


        img_in = np.reshape(img_rsz,[1,160,160,3])

        b_c = bearing_compensation([p_lat,p_lon],[lat,lon])

        bearings[p] = (sess.run(logits, feed_dict={image: img_in})[0][0] - ang_diff + b_c)




    bearing_avg = np.average(bearings)

    bearing = bearing_avg

    
    
    db_heading = float(points[2][0][5])
    ang_diff_abs = np.minimum(np.abs(db_heading - heading_2_dest), 360 - np.abs(db_heading - heading_2_dest))
    db_h_shift = (db_heading + (360 - heading_2_dest))%360
    ang_diff_sign = np.sign(db_h_shift - 180)
    ang_diff = ang_diff_abs * ang_diff_sign

    print('cord: ',points[2][0][0],',',points[2][0][1],' heading: ',db_heading,', heading to dest: ',heading_2_dest,'headings diff: ',ang_diff, 'correction: ',bearing)
    print('done')


#---------------------------------------------------------------------------------------------------------
def location_360_test():
  size = 8 
 # test_points=[[31.630185, 34.821536],[31.64212421,34.79050406],[31.6483181,34.7808723],[31.6483181,34.7808723],[31.65319346,34.76873743],[31.66124538,34.7553989],[31.66512506,34.74895476]]
  test_points=[[31.630185, 34.821536]]

  with tf.Graph().as_default() as g:
    image = tf.placeholder(dtype=tf.float32,shape=(1,160,160,3))      
    
    logits = graph['s']
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)   
    init = tf.global_variables_initializer()
    sess.run(init, {is_training: False})
    saver = tf.train.Saver(graph['v'])
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg19_train_holes/')
    saver.restore(sess, ckpt)
    idx = 0
    for lat,lon in test_points:
      idx = idx + 1
      f = open("loaction test" + str(idx) + " 360.txt", "a")
      heading_2_dest = HeadingTo ([lat, lon],[31.669243,34.742072])

      start = 82
      for heading in range(start,360):
        points = find_closest_points(lat, lon,heading,size)
        if (points[0] == []):
          raise 'did not find point'

        bearings = [[]]*len(points)
        locations = [[]]*len(points)
        for p in range(len(points)):
          if (points[p] != []):
            ang_diff = points[p][2]
            file_name = points[p][0][3]
            frame_num = int(points[p][0][4])
            p_lat = float(points[p][0][0])
            p_lon = float(points[p][0][1])
            locations[p] = [p_lat,p_lon]
            if (file_name in ['33','34','35', '46', '47', '50', '51_1', '53', '54']):
              filetype = '.mp4'
            else:
              filetype = '.mov'

            filename = 'E:/Final Project/walk/vids/DJI_00' + file_name + filetype   
            vid = imageio.get_reader(filename,'ffmpeg')

            img = vid.get_data(frame_num)
            img_rsz = imresize(img,[160,160,3])
            img_rsz[:,:,1] = 0
            if (p == 2):
              print('vid: '+ str(file_name) + ', frame num: '+ str(frame_num) + ', filetype: ' + filetype)


            img_in = np.reshape(img_rsz,[1,160,160,3])

            b_c = bearing_compensation([p_lat,p_lon],[lat,lon])

            bearings[p] = (sess.run(logits, feed_dict={image: img_in})[0][0] - ang_diff + b_c)

        bearing_avg = np.average(bearings)
        bearing = bearing_avg
 
    
        db_heading = float(points[2][0][5])
        ang_diff_abs = np.minimum(np.abs(db_heading - heading_2_dest), 360 - np.abs(db_heading - heading_2_dest))
        db_h_shift = (db_heading + (360 - heading_2_dest))%360
        ang_diff_sign = np.sign(db_h_shift - 180)
        ang_diff = ang_diff_abs * ang_diff_sign
    
        error = np.minimum(np.abs(bearing - ang_diff), 360 - np.abs(bearing - ang_diff))
        print('file: ',idx,' heading: ',db_heading,'cord: ',points[2][0][0],',',points[2][0][1], 'heading to dest: ',heading_2_dest,'headings diff: ',ang_diff, 'correction: ',bearing,'error: ',error)
        f.write(str(db_heading) + ',' + points[2][0][0]+','+points[2][0][1]+','+str(heading_2_dest)+','+str(ang_diff)+','+str(bearing)+','+str(error)+'\n')
    
      f.close()        
      print(str(idx) + ' done')


def neurons_test():
  size = 3 
  test_points=[[31.630185, 34.821536],[31.64212421,34.79050406],[31.6483181,34.7808723],[31.6483181,34.7808723],[31.65319346,34.76873743],[31.66124538,34.7553989],[31.66512506,34.74895476]]
  #test_points=[[31.630185, 34.821536]]
  thresh = 5
  with tf.Graph().as_default() as g:
    image = tf.compat.v1.placeholder(dtype=tf.float32,shape=(1,160,160,3))      
    graph = vgg.inference(image, gpu,tf.constant(False))
    activations = graph['a']
    act_thr = {}
    for layer in activations:
      if 'fc' in layer:
        act_thr[layer] = np.zeros([360,activations[layer].shape[1]])
   
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)   
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(graph['v'])
    ckpt = tf.train.latest_checkpoint('D:/tensorflow_fs/vgg_19_train_test/')
    saver.restore(sess, ckpt)
    idx = 0
    for lat,lon in test_points:
      idx = idx + 1      
      heading_2_dest = HeadingTo ([lat, lon],[31.669243,34.742072])

      start = 1
      for heading in range(start,360):
     
        points = find_closest_points2(lat, lon,heading)
        if (points[0] == []):
          raise 'did not find point'

        bearings = [[]]*len(points)
        locations = [[]]*len(points)
        for p in range(len(points)):
          if (points[p] != []):
            ang_diff = points[p][2]
            file_name = points[p][0][3]
            frame_num = int(points[p][0][4])
            p_lat = float(points[p][0][0])
            p_lon = float(points[p][0][1])
            locations[p] = [p_lat,p_lon]
            p_heading = int(np.floor(float(points[p][0][5])))

            if (file_name in ['33','34','35', '46', '47', '50', '51_1', '53', '54']):
              filetype = '.mp4'
            else:
              filetype = '.mov'

            filename = 'E:/Final Project/walk/vids/DJI_00' + file_name + filetype   
            vid = imageio.get_reader(filename,'ffmpeg')

            img = vid.get_data(frame_num)
            img_rsz = imresize(img,[160,160,3])
            img_rsz[:,:,1] = 0
            if (p == 1):
              print('vid: '+ str(file_name) + ', frame num: '+ str(frame_num) + ', filetype: ' + filetype)

            img_in = np.reshape(img_rsz,[1,160,160,3])         
            for layer in activations:
              if 'fc' in layer:
                act_thr[layer][p_heading] += (sess.run(activations[layer],feed_dict={image: img_in})[0])/7           
        
    for layer in activations:
      if 'fc' in layer:
        scipy.io.savemat('E:/Final Project/walk/neurons/'+layer+'.mat', mdict={layer: act_thr[layer]})        
    print(str(idx) + ' done')  


def main(argv=None):  # pylint: disable=unused-argument
  #walk()
  batch_walk()
  #batch_test()
  #location_test()
  #location_360_test()
  #image_test()
  #image_test_custom()
  #image_list_test()
  #cnn_vis()
  #neurons_test()


if __name__ == '__main__':
  tf.app.run()
