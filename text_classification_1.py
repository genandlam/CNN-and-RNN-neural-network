from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import csv
import time
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from scipy.misc import toimage


#from io import open
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#K.set_session(sess)
#print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

#K.set_image_dim_ordering('tf')

seed = 10
np.random.seed(seed)

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
BATCH_SIZE=128
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x,drop_out):
  
  input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
   
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    
    if drop_out:
        print("yes")
        pool1 = tf.nn.dropout(pool1, keep_prob=0.8)
        
    conv2 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
            
    pool2 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    
    if drop_out:
          pool2 = tf.nn.dropout(pool2, keep_prob=0.8)
    
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return input_layer, logits
  
def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pd.Series(x_train)
  y_train = pd.Series(y_train)
  x_test = pd.Series(x_test)
  y_test = pd.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test


def main(drop_out):
  
  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = char_cnn_model(x,drop_out)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), y_)  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  train_loss = []
  test_loss = []
  train_accuracy = []
  test_accuracy = []
  t1 = time.time()
  # training
  for e in range(no_epochs):
        
    # mini-batch training
    for batch in range(len(x_train)//BATCH_SIZE):
            
            batch_x = x_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(x_train))]
            batch_y = y_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(y_train))] 
            
            opt = sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
                                                              
            
            loss, acc = sess.run([entropy, accuracy], feed_dict={x: batch_x, y_: batch_y})
                                                              
            
#             _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
#             loss.append(loss_)

#            if e%1 == 0:
#              print('iter: {}, entropy: {:.3f}, acc: {:.3f}, test_acc: {:.3f}'.format(e, train_loss[e],
#                                                                                train_acc[e], test_acc[e]))
    
    print("Iter " + str(e) + ", entropy= " + \
                      "{:.3f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    print("Optimization Finished!")

    # Calculate accuracy for all 10000 mnist test images
    test_acc,valid_loss = sess.run([accuracy,entropy], feed_dict={x: x_test,y_ : y_test})
    train_loss.append(loss)
    test_loss.append(valid_loss)
    train_accuracy.append(acc)
    test_accuracy.append(test_acc)
    print("Testing Accuracy:","{:.5f}".format(test_acc))
  print("Used time,100 epochs: {}".format(time.time() - t1))  
#    sess.close()
#  plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
#  plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
  plt.plot(range(len(train_loss)), train_loss, label='Training loss')
  plt.plot(range(len(test_accuracy)), test_accuracy,  label='Test accuracy')
  plt.plot(range(len(train_accuracy)), train_accuracy,  label='Traing accuracy')
  plt.title('Training loss and Test accuracy')
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss/Accuracy',fontsize=16)
  plt.legend()
#  plt.figure()
#  plt.savefig("filename.png")  
#  plt.savefig('./p2a_1_dropout.png')  
#  plt.show()

#  plt.plot(range(len(total_train_acc)), total_train_acc, label='train_acc')
#  plt.plot(range(len(total_train_loss)), total_train_loss, label='train_loss')
#  plt.plot(range(len(total_test_acc)), total_test_acc, label='test acc')
#  plt.legend()
#  plt.title('Accuracy/loss of model')
#  plt.xlabel('Epochs')
#  plt.ylabel('Accuracy/loss')

if __name__ == '__main__':
    
  main(False)
   #x_train, y_train, x_test, y_test = read_data_chars()
   

  
  

