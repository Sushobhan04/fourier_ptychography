from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    Cropping2D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping
import math
import math
import sys
import tensorflow as tf 

K.set_image_dim_ordering('th')

if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1),border_mode = 'same'):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode=border_mode)(input)
        # conv = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(conv)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual,border_mode = 'same'):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    # stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    # stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    # equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]
    # N = tf.shape(input)[2]
    # M = tf.shape(residual)[2]

    if border_mode=='valid':
        shortcut = Cropping2D(cropping=((4,4),(4,4)))(input)
    else:
        shortcut = input
    # shortcut = input
    # # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    #     shortcut = Convolution2D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
    #                              nb_row=1, nb_col=1,
    #                              subsample=(stride_width, stride_height),
    #                              init="he_normal", border_mode="same")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_subsample=(1, 1), kernel_shape = (5,5), border_mode = 'same'):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, kernel_shape[0],kernel_shape[1], subsample=init_subsample, border_mode = border_mode)(input)
        residual = _conv_bn_relu(nb_filters, kernel_shape[0],kernel_shape[1], border_mode = border_mode)(conv1)
        return _shortcut(input, residual, border_mode)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f

def crop_tensor(set1,set2):
    pass

def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]

def BatchGenerator(files,batch_size,border_mode):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data['data'])
            data = -data + 1.0
            label = np.array(curr_data['label'])
            label - -label + 1.0
            if border_mode=='valid':
                label = crop(label,10)
            # print curr_data
            # print file
            # print data.shape
            for i in range(data.shape[0]//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, label_bat)

def schedule(epoch):
    if epoch<40:
        return 0.0001
    else:
        return 0.0001

def train_model(path_train,home,model_name,mParam):

    # path_train = '/home/sushobhan/caffe/data/ptychography/databases/Test42_Set91_img512_patch48/train_images/'
    # path_train =  "/home/sushobhan/Documents/data/ptychography/Test42_Set91_img512_patch48/train_images/"
    # model_spec = sys.argv[1]

    file = h5py.File(path_train+'set_1.h5','r')
    ks = file.keys()
    # print ks

    data = file['data']
    label = file['label']

    print data.shape, label.shape
    print np.max(data), np.max(label)


    input_shape = data.shape[1:]
    output_shape = label.shape[1:]

    print input_shape,output_shape

    # input_shape = (49,None,None)
    # output_shape = (1,None,None)
    print input_shape,output_shape

    border_mode = mParam['border_mode']
    input = Input(shape=input_shape)

    temp = Convolution2D(128, 5, 5, border_mode=border_mode, init = 'he_normal')(input)
    # temp = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(temp)
    temp = Activation('relu')(temp)

    temp = basic_block(128, kernel_shape = (3,3),border_mode = border_mode)(temp)

    # temp = basic_block(32, kernel_shape = (5,5))(temp)

    temp = Convolution2D(1, 3, 3, border_mode=border_mode, init = 'he_normal')(temp)
    # temp = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(temp)
    temp = Activation('relu')(temp)
    

    model = Model(input=input, output=temp)

    lrate = mParam['lrate']
    epochs = mParam['epochs']
    decay = mParam['decay']
    train_batch_size = mParam['train_batch_size']
    val_batch_size = mParam['val_batch_size']
    samples_per_epoch = mParam['sam_epoch']
    # sgd = Adadelta(lr=lrate, rho=0.95, epsilon=1e-08, decay=decay)

    train_files = []
    val_files = []
    for i in range(1,5):
        train_files.append(path_train+'set_'+str(i)+'.h5')

    for i in range(5,6):
        val_files.append(path_train+'set_'+str(i)+'.h5')

    # print files
    train_generator = BatchGenerator(train_files,train_batch_size,border_mode)
    val_generator = BatchGenerator(val_files,val_batch_size,border_mode)
    lrate = LearningRateScheduler(schedule)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    callbacks_list = [lrate,early_stop]

    sgd = SGD(lr=0.0, momentum=0.9, decay=decay, nesterov=True)

    model.compile(loss='mean_squared_error',
              optimizer=sgd)

    model.fit_generator(train_generator,validation_data=val_generator,nb_val_samples=1000, samples_per_epoch = samples_per_epoch, nb_epoch = epochs,verbose=1 ,callbacks=callbacks_list)

    model.save(home+'models/'+model_name+'.h5')

    # print model.summary()
def main():
    path_train =  "/home/sushobhan/Documents/data/ptychography/Test42_Set91_img512_patch48/train_images/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]

    mParam = {}
    mParam['lrate'] = 0.0001
    mParam['epochs'] = 40
    mParam['decay'] = 0.0
    mParam['train_batch_size'] = 512
    mParam['sam_epoch'] = 10000
    mParam['val_batch_size'] = 1000
    mParam['border_mode'] = 'same'

    train_model(path_train,home,model_name,mParam)

if __name__ == '__main__':
    main()