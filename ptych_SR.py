from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
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
import sys

K.set_image_dim_ordering('th')

if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]

def BatchGenerator(files,batch_size,border_mode,crop_size):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data['data'])
            label = np.array(curr_data['label'])
            # data = np.max(data) - data
            # label = np.max(label) - label
            # label - -label + 1.0
            if border_mode=='valid':
                label = crop(label,crop_size)
            # print curr_data
            # print file
            # print data.shape
            for i in range(data.shape[0]//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, label_bat)

def schedule(epoch):
    lr = 0.001
    if epoch<30:
        return lr
    elif epoch<300:
        return lr/4
    elif epoch<800:
        return lr/4
    else:
        return lr/8

def _residual_block(block_function, nb_filter, kernel_size, repetitions):
    def f(input):
        res = input
        for i in range(repetitions):
            res = _basic_conv_block(nb_filter=nb_filter,kernel_size=kernel_size)(res)

        res = _conv_relu(nb_filter=nb_filter,kernel_size=kernel_size)(res)

        res = merge.Add([res, input])
        return res

    return f

def _basic_conv_block(nb_filter, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, (kernel_size,kernel_size), subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f

def _basic_deconv_block(nb_filter, kernel_size = 3, subsample=(2, 2)):
    def f(input):
        conv = Conv2DTranspose(nb_filter=nb_filter, (kernel_size,kernel_size), subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f



def _conv_relu(nb_filter, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, (kernel_size,kernel_size), subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return norm

    return f


def train_model(path_train,home,model_name,mParam):

    input_shape = (49,None,None)
    output_shape = (1,None,None)
    print input_shape,output_shape

    border_mode = mParam['border_mode']
    norm_axis = 1

    input = Input(shape=input_shape)

    kernels = [5,3,3]

    temp = Convolution2D(128, kernels[0], kernels[0], border_mode=border_mode, init = 'he_normal')(input)
    # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    temp = Activation('relu')(temp)

    # temp = Convolution2D(64, 5, 5, border_mode=border_mode, init = 'he_normal')(temp)
    # # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    # temp = Activation('relu')(temp)

    # temp = Convolution2D(128, 5, 5, border_mode=border_mode, init = 'he_normal')(temp)
    # # temp = Dropout(0.5)(temp)
    # # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    # temp = Activation('relu')(temp)
    
    temp = Convolution2D(128, kernels[1], kernels[1], border_mode=border_mode, init = 'he_normal')(temp)
    # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    temp = Activation('relu')(temp)
    

    temp = Convolution2D(1, kernels[2], kernels[2], border_mode=border_mode, init = 'he_normal')(temp)
    # temp = BatchNormalization(mode=1, axis=norm_axis)(temp)
    # temp = Activation('relu')(temp)

    model = Model(input=input, output=temp)

    crop_size = (sum(kernels) - len(kernels))//2

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
    train_generator = BatchGenerator(train_files,train_batch_size,border_mode,crop_size)
    val_generator = BatchGenerator(val_files,val_batch_size,border_mode,crop_size)
    lrate_sch = LearningRateScheduler(schedule)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    callbacks_list = [lrate_sch,early_stop]

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)

    model.compile(loss='mean_squared_error',
              optimizer=sgd)

    model.fit_generator(train_generator,validation_data=val_generator,nb_val_samples=1000, samples_per_epoch = samples_per_epoch, nb_epoch = epochs,verbose=1 ,callbacks=callbacks_list)

    model.save(home+'models/'+model_name+'.h5')

    # print model.summary()

def main():
    path_train =  "/home/sushobhan/Documents/data/fourier_ptychography/train_images/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]

    mParam = {}
    mParam['lrate'] = 0.001
    mParam['epochs'] = 40
    mParam['decay'] = 0.0
    mParam['train_batch_size'] = 512
    mParam['sam_epoch'] = 10944
    mParam['val_batch_size'] = 1000
    mParam['border_mode'] = 'valid'

    train_model(path_train,home,model_name,mParam)


if __name__ == '__main__':
    main()