from __future__ import print_function
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from scipy.misc import toimage


os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

import numpy as np
import pandas
import tensorflow as tf
import csv

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
BATCH_SIZE = 128

no_epochs = 20
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x,n_words,drop_out):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  num_layers=2  

  with tf.variable_scope("RNN") as scope:
    cells=[]
    for _ in range(num_layers):  
        
      cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE) 

      if drop_out:
          cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.9, output_keep_prob=0.9) 
        
      cells.append(cell)
    
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    output, state = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    logits = tf.layers.dense(output[-1], MAX_LABEL, activation=None)

  return logits,word_list

def data_read_words():
  
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
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
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
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, word_list = rnn_model(x,n_words,drop_out)

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

  # training
#  for e in range(no_epochs):
#     _, loss_  = sess.run([ train_op, entropy], {x: x_train, y_: y_train})
#    loss.append(loss_)

#    if e%10 == 0:
#      print('epoch: %d, entropy: %g'%(e, loss[e]))
  for e in range(no_epochs):
    # mini-batch training
    for batch in range(len(x_train)//BATCH_SIZE):
            
            batch_x = x_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(x_train))]
            batch_y = y_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(y_train))] 
              # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            
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
   
    print("Testing Accuracy:","{:.5f}".format(test_acc))
    
  plt.plot(range(len(train_loss)), train_loss, label='Training loss')
  plt.plot(range(len(test_accuracy)), test_accuracy,  label='Test accuracy')
  plt.title('Training loss and Test accuracy')
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss/Accuracy',fontsize=16)
  plt.legend()
#  plt.figure()
  plt.savefig('./p2_6b_2.png') 
  
if __name__ == '__main__':
  main(False)
   # x_train, y_train, x_test, y_test,no_word = data_read_words()
