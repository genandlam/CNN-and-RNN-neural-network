from __future__ import print_function
import os
import numpy as np
import sklearn
#from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import csv

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from scipy.misc import toimage
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#K.set_session(sess)
#print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

#K.set_image_dim_ordering('tf')

seed = 10
np.random.seed(seed)

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
BATCH_SIZE=128
no_epochs = 20
lr = 0.01
embedding_size=20

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_cnn_model(x,no_words,drop_out):
    
  word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_words, embed_dim=embedding_size)

  input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, embedding_size, 1])

  with tf.variable_scope('CNN_Layer2'):
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
            pool1 = tf.nn.dropout(pool1, keep_prob=0.9)
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
            pool2 = tf.nn.dropout(pool2, keep_prob=0.9)
    
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return input_layer, logits
  
def read_data_word():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pd.Series(x_train)
  y_train = pd.Series(y_train)
  x_test = pd.Series(x_test)
  y_test = pd.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words) 
  
  return x_train, y_train, x_test, y_test, no_words


def main(drop_out):
  
  x_train, y_train, x_test, y_test,no_words = read_data_word()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  
  
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

#  inputs, logits = word_cnn_model(x)
  inputs, logits = word_cnn_model(x, no_words,drop_out)
  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), y_)  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
  
  sess = tf.Session(config=config)
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
                                                              
            

    
    print("Iter " + str(e) + ", entropy= " + \
                      "{:.3f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
    print("Optimization Finished!")

    # Calculate accuracy for all 
    test_acc,valid_loss = sess.run([accuracy,entropy], feed_dict={x: x_test, y_ : y_test})

    train_loss.append(loss)
    test_loss.append(valid_loss)
    train_accuracy.append(acc)
    test_accuracy.append(test_acc)
    
#    if e%1 == 0:
#              print('iter: {}, entropy: {:.3f}, acc: {:.3f}, test_acc: {:.3f}'.format(e, loss,                                                                                acc, test_acc))
    print("Testing Accuracy:","{:.5f}".format(test_acc))
  

  print("Used time,20 epochs: {}".format(time.time() - t1))  
    
#  plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
#  plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
  plt.plot(range(len(train_loss)), train_loss, label='Training loss')
  plt.plot(range(len(test_accuracy)), test_accuracy,  label='Test accuracy')
  plt.title('Training loss and Test accuracy')
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss/Accuracy',fontsize=16)
  plt.legend()
#  plt.figure()
#  plt.savefig('./p2b_2_drop_out.png')  
#  plt.show()

if __name__ == '__main__':
   main(False)
   # x_train, y_train, x_test, y_test,no_word = read_data_word()

  
  

