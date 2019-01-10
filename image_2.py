rom __future__ import print_function
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
epochs = 355
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

print(X_train.shape[0], 'train samples')

# chans, # images, # rows, # cols)
X_train = X_train.reshape(X_train.shape[0],IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0],IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

print(X_train.shape)
print(X_test.shape)

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

def cnn(X_train, y_train,X_test, y_test,batch_size,epochs,NUM_CLASSES,
       IMG_SIZE, NUM_CHANNELS,learning_rate,con_size,kernel_size,
       con_size2,kernel_size2 ):
        model = Sequential()
        model.add(Conv2D(con_size, kernel_size=kernel_size,padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS),               activation='relu',name='con1'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2 ,name='pool1'))
        model.add(Conv2D(con_size2,kernel_size=kernel_size2 ,padding='valid', activation='relu',name='con2'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,name='pool2'))
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        #decay = lrate/epochs
        sgd =  optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print(model.summary())

        history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=epochs, batch_size=batch_size,callbacks=[plot_losses])
        # Final evaluation of the model

        score_test = model.evaluate(X_test, y_test, verbose=0)
        print('Test accuracy:', score_test[1])
        return  model, score_test[1], history

max_acc=0
model_par=[]
CON_SIZE=[32,64,128] 
CON_SIZE2=[32,64,128]
KERNAL_SIZE=[2,3,5]
KERNAL_SIZE2=[2,3,5]
EPOCHS=[100,200,500]

for con_size in CON_SIZE:
    for kernel_size in KERNAL_SIZE:
        for con_size2 in CON_SIZE2:
            for kernel_size2 in KERNAL_SIZE2:
                for epochs in EPOCHS:
                    
                    test_accuracy,model= cnn(X_train, y_train,X_test, y_test,batch_size,epochs,NUM_CLASSES,
                    IMG_SIZE, NUM_CHANNELS,learning_rate,con_size,kernel_size,con_size2,kernel_size2)
                    if max_acc < test_accuracy:
                        max_model=model
                        max_acc=test_accuracy
                        model_par.append('test_accuracy: '+str(test_accuracy)+' CON_SIZE: '+str(con_size) +'kernel_size:'+str(kernel_size)+' CON_SIZE2:'+str(con_size2)+' kernel_size2: '+str(kernel_size2)+ ' epochs:'+str(epochs))
                        
def plot_feature_maps(feature_maps,layer_name):
    height, width, depth = feature_maps.shape
    nb_plot = int(np.rint(np.sqrt(depth)))
    
    fig = plt.figure(figsize=(10, 10))
    for i in range(depth-7):
        plt.axis('off')
        plt.subplot(nb_plot, nb_plot, i+1)
        plt.imshow(feature_maps[:,:,i],cmap='gray')
        #plt.title('feature map {}'.format(i+1))
#    plt.savefig('./p1b_2_pool2.png')
#    plt.savefig('./p1_2_'+layer_name+'.png')
    plt.show()
    
#plot_feature_maps(pool1_featres[0])



def plot_layer(layer_name):
    # check if the layer_name is correct 
    X = X_test[2,:]
    X=np.asarray(X)

    #X_test = X_test.reshape(X_test.shape[0],IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    X=X.reshape(1,IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    
    features_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = features_extractor.predict(X)[0]
    print("At layer \"{}\" : {} ".format(layer_name, feature_maps.shape))
    plot_feature_maps(feature_maps,layer_name)
    
plot_layer('con1')
plot_layer('con2')
plot_layer('pool1')
plot_layer('pool2')


