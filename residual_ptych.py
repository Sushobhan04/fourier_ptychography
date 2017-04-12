from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Reshape
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose
)
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping
import math
import sys
from keras import losses

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

def BatchGenerator(files,batch_size,dtype = 'train'):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data[dtype]['data'][()])
            label = np.array(curr_data[dtype]['label'][()])
            # print data.shape, label.shape

            for i in range((data.shape[0]-1)//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, label_bat)

def TrainingSetGenerator(file):
    curr_data = h5py.File(file,'r')
    data = np.array(curr_data['data'])
    label = np.array(curr_data['label'])
    print data.shape, label.shape
    return data,label

def schedule(epoch):
    lr = 0.00001
    if epoch<15:
        return lr
    elif epoch<150:
        return lr/10
    elif epoch<300:
        return lr/40
    elif epoch<450:
        return lr/160
    else:
        return lr/1000

def _residual_block(filters, kernel_size = 3, repetitions=1):
    def f(input):
        res = input
        for i in range(repetitions):
            res = _conv_relu(filters=filters,kernel_size=kernel_size)(res)

        res = _conv(filters=filters,kernel_size=kernel_size)(res)

        res = merge([res, input], mode = 'sum')
        return res

    return f

def _conv_block(filters, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer="he_normal", padding="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f

def _dense_block(node_arr):
    def f(input):
        temp = input
        for nodes in node_arr:
            temp = Dense(nodes,kernel_initializer='he_normal')(temp)
            temp = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(temp)
            temp = Activation("relu")(temp)
        return temp

    return f

def _deconv_block(filters, kernel_size = 3, strides=(2, 2)):
    def f(input):
        conv = Conv2DTranspose(filters=filters, kernel_size = (kernel_size,kernel_size), strides=subsample,
                             kernel_initializer="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f



def _conv_relu(filters, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer="he_normal", border_mode="same")(input)
        active = Activation("relu")(conv)
        return active

    return f

def _conv_bn(filters, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return norm

    return f

def _conv(filters, kernel_size = 3, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer="he_normal", border_mode="same")(input)
        return conv

    return f

def create_model(input_shape, output_shape):
    input = Input(shape=input_shape)

    temp = _conv_relu(filters = 64, kernel_size = 5)(input)

    temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1)(temp)
    temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1)(temp)
    temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1)(temp)
    temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1)(temp)
    temp = _conv(filters = 1, kernel_size = 5)(temp)

    # temp = concatenate([temp1,temp2],axis=CHANNEL_AXIS)


    model = Model(input= input, output = temp)

    return model


def train_model(path_train,home,model_name,mParam):

    lrate = mParam['lrate']
    epochs = mParam['epochs']
    decay = mParam['decay']
    train_batch_size = mParam['train_batch_size']
    val_batch_size = mParam['val_batch_size']
    steps_per_epoch = mParam['steps_per_epoch']
    validation_steps = mParam['validation_steps']

    input_shape = mParam['input_shape']
    output_shape = mParam['output_shape']

    print input_shape,output_shape

    border_mode = mParam['border_mode']
    norm_axis = 1

    model = create_model(input_shape, output_shape)

    # train_files = [path_train+'data/'+'dataset_1.h5']
    # val_files = [path_train+'data/'+'valset_1.h5']
    dataset = []

    for i in range(1,6):
        dataset.append(path_train+'datasets/pcp_ptych/'+'set_'+str(i)+'.h5')

    train_generator = BatchGenerator(dataset,train_batch_size,dtype = 'train')
    val_generator = BatchGenerator(dataset,val_batch_size,dtype = 'val')
    lrate_sch = LearningRateScheduler(schedule)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')
    callbacks_list = [lrate_sch,early_stop]

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)

    # model.compile(loss='mean_squared_error',
    #           optimizer=sgd)

    model.compile(loss='mean_squared_error',
              optimizer=sgd)

    # print validation_steps

    model.fit_generator(train_generator,validation_data=val_generator, validation_steps = validation_steps,steps_per_epoch = steps_per_epoch, epochs = epochs,verbose=1 ,callbacks=callbacks_list)
    # model.fit(data, label, batch_size=train_batch_size, nb_epoch=epochs, verbose=1, callbacks=callbacks_list, validation_split=0.1, shuffle=True)
    model.save(path_train+'models/'+model_name+'.h5')

    # print model.summary()


def main():

    path_train =  "/home/sushobhan/Documents/data/fourier_ptychography/"
    home = "/home/sushobhan/Documents/research/fourier_ptychography/"
    # set_name = sys.argv[1]
    model_name = sys.argv[1]

    N = 64

    mParam = {}
    mParam['lrate'] = 0.001
    mParam['epochs'] = 20
    mParam['decay'] = 0.0
    mParam['border_mode'] = 'same'

    mParam['input_shape'] = (49,None, None)
    mParam['output_shape'] = (1, None, None)

    mParam['train_batch_size'] = 64
    mParam['val_batch_size'] = 64
    mParam['steps_per_epoch'] = 12000//mParam['train_batch_size']
    mParam['validation_steps'] = 1500//mParam['val_batch_size']
    # mParam['set_name'] = set_name

    train_model(path_train,home,model_name,mParam)


if __name__ == '__main__':
    main()