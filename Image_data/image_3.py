from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten,\
                         Dropout,GlobalMaxPooling1D, Lambda, Concatenate, Dense, regularizers
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers, activations

from keras import callbacks
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

from matplotlib import pyplot as plt
from IPython.display import clear_output
from scipy.misc import toimage
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="1"


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

K.set_image_dim_ordering('tf')

seed = 10
np.random.seed(seed)

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 450
batch_size = 128

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    print(labels_.shape)
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

file_name=('data_batch_1')
image_train, y_train=load_data(file_name)
image_test, y_test=load_data('test_batch')
print(image_train.shape)

X_train = image_train / 255.0
X_test = image_test / 255.0

# Convert class vectors to binary class matrices.

print(y_train.shape)

# chans, # images, # rows, # cols)
X_train = X_train.reshape(X_train.shape[0],IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0],IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

print(X_train.shape)
print(X_test.shape)

model = Sequential()
### Remember to change to the optimal version
model.add(Conv2D(32, (2, 2),padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), activation='relu',name='con1'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
model.add(Conv2D(128,(3,3) ,padding='valid', activation='relu',name='con2'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#decay = lrate/epochs
sgd =  optimizers.SGD(lr=learning_rate, momentum=0.1, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
         
class PlotLosses(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.legend()
        
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
#        plt.plot(self.x, self.val_losses, label="val_loss")
        
        plt.show();
        
plot_losses = PlotLosses()

def train_loss(name,history):
    #training cost 
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_cost'], loc='upper left')
    plt.savefig('./p1_3'+name+'_train_cost_.png')
    plt.show()

def test_acc(name,history):
    # summarize history for accuracy
    # test accuracy 
    #plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['test_acc'], loc='upper left')
    plt.savefig('./p1_3'+name+'_test_acc_.png')
    plt.show()
    # summarize history for loss

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
# Final evaluation of the model

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])


train_loss('a',history)
test_acc('a',history)


model = Sequential()
model.add(Conv2D(32, (2, 2),padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), activation='relu',name='con1'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
model.add(Conv2D(128,(3,3) ,padding='valid', activation='relu',name='con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#decay = lrate/epochs
#sgd =  optimizers.SGD(lr=learning_rate, momentum=0.1, decay=0.0, nesterov=False)
#optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
         
epochs = 15

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
# Final evaluation of the model

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])
his2=history

train_loss('b',his2)
test_acc('b',his2)


model = Sequential()
model.add(Conv2D(32, (2, 2),padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), activation='relu',name='con1'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
model.add(Conv2D(128,(3,3) ,padding='valid', activation='relu',name='con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#decay = lrate/epochs
#sgd =  optimizers.SGD(lr=learning_rate, momentum=0.1, decay=0.0, nesterov=False)
rmsprop=optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
#adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
print(model.summary())
         
epochs = 95

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
# Final evaluation of the model

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])
his3=history

train_loss('c',his3)
test_acc('c',his3)

model = Sequential()
model.add(Conv2D(32, (2, 2),padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), activation='relu',name='con1'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
model.add(Conv2D(128,(3,3) ,padding='valid', activation='relu',name='con2'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#decay = lrate/epochs
#sgd =  optimizers.SGD(lr=learning_rate, momentum=0.1, decay=0.0, nesterov=False)
#rmsprop=optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
         
epochs = 30

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
# Final evaluation of the model

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])

model = Sequential()
model.add(Conv2D(32, (2, 2),padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), activation='relu',name='con1'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
model.add(Conv2D(128,(3,3) ,padding='valid', activation='relu',name='con2'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#decay = lrate/epochs
#sgd =  optimizers.SGD(lr=learning_rate, momentum=0.1, decay=0.0, nesterov=False)
#rmsprop=optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
         
epochs = 30

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
# Final evaluation of the model

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])
his4=history

train_loss('d',his4)
test_acc('d',his4)






