from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    Reshape
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Cropping2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import (Add, Multiply)
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta, Adam
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint
import math
import sys
from keras import losses
from keras import regularizers

K.set_image_dim_ordering('th')

lrate = 0.00001

kernel_initializer = 'he_uniform'
bias_initializer = "zeros"
kernel_regularizer=regularizers.l2(0.0005)
# activity_regularizer=regularizers.l1(0.01)

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

def normalize(arr):
    ma = np.max(arr,axis=(2,3))
    mi = np.min(arr,axis=(2,3))
    ma = np.expand_dims(np.expand_dims(ma,axis=2),axis=3)
    mi = np.expand_dims(np.expand_dims(mi,axis=2),axis=3)

    arr = np.divide((arr - mi),(ma-mi))

    return arr




def BatchGenerator(files,batch_size, N = 0):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data['data'][()])
            label = np.array(curr_data['label'][()])
            # print data.shape, label.shape

            for i in range((data.shape[0]-1)//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,24:25,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                label_bat = label_bat**2
                # yield (normalize(data_bat), crop(label_bat,N))
                yield (data_bat, crop(label_bat,N))

def TrainingSetGenerator(file):
    curr_data = h5py.File(file,'r')
    data = np.array(curr_data['data'])
    label = np.array(curr_data['label'])
    print data.shape, label.shape
    return data,label

def schedule(epoch):
    lr = lrate
    if epoch<50:
        return lr
    elif epoch<200:
        return lr/10
    elif epoch<400:
        return lr/50
    elif epoch<450:
        return lr/160
    else:
        return lr/1000

def _residual_block(filters, kernel_size = 3, repetitions=1, padding = 'same'):
    def f(input):
        res1 = input
        for i in range(repetitions):
            res1 = _conv_relu(filters=filters,kernel_size=kernel_size, padding = padding)(res1)

        res1 = _conv(filters=filters,kernel_size=kernel_size, padding = padding)(res1)
        if padding=='valid':
            res2 = Cropping2D(cropping = ((kernel_size-1)//2)*(repetitions+1))(input)
        else:
            res2 = input

        res = Add()([res1, res2])

        return res

    return f

def _conv_block(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f

def _dense_block(node_arr):
    def f(input):
        temp = input
        for nodes in node_arr:
            temp = Dense(nodes,kernel_initializer=kernel_initializer,kernel_regularizer = kernel_regularizer)(temp)
            temp = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(temp)
            temp = Activation("relu")(temp)
        return temp

    return f

def _deconv_block(filters, kernel_size = 3, strides=(2, 2), padding = 'same'):
    def f(input):
        conv = Conv2DTranspose(filters=filters, kernel_size = (kernel_size,kernel_size), strides=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer, 
                             kernel_regularizer = kernel_regularizer,padding = padding)(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f



def _conv_relu(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        active = Activation("relu")(conv)
        return active

    return f

def _conv_bn(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return norm

    return f

def _conv(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        return conv

    return f

def create_residual_model(input_shape, output_shape, padding = 'same'):
    input = Input(shape=input_shape)

    temp = _conv_relu(filters = 64, kernel_size = 9, padding = padding)(input)

    temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1, padding = padding)(temp)
    # temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1, padding = padding)(temp)
    # temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1, padding = padding)(temp)
    # temp = _residual_block(filters = 64, kernel_size = 3, repetitions=1, padding = padding)(temp)
    temp = _conv(filters = 1, kernel_size = 5, padding = padding)(temp)

    # temp = concatenate([temp1,temp2],axis=CHANNEL_AXIS)
    temp = Multiply()([temp,temp])


    model = Model(input= input, output = temp)

    return model

def create_conv_model(input_shape, output_shape, padding = 'same'):
    input = Input(shape=input_shape)

    temp = _conv_relu(filters = 32, kernel_size = 9, padding = padding)(input)
    temp = _conv_relu(filters = 16, kernel_size = 5, padding = padding)(temp)
    temp = _conv(filters = 1, kernel_size = 5, padding = padding)(temp)

    # temp = concatenate([temp1,temp2],axis=CHANNEL_AXIS)
    temp = Multiply()([temp,temp])


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
    crop = mParam['crop']

    print input_shape,output_shape

    padding = mParam['border_mode']
    norm_axis = 1

    model = create_conv_model(input_shape, output_shape, padding = padding)

    # train_files = [path_train+'data/'+'dataset_1.h5']
    # val_files = [path_train+'data/'+'valset_1.h5']
    train_dataset = []
    val_dataset = []

    for i in range(1,6):
        train_dataset.append(path_train+'datasets/Test42/train/'+'set_'+str(i)+'.h5')

    for i in range(1,2):
        val_dataset.append(path_train+'datasets/Test42/test/'+'set_'+str(i)+'.h5')

    train_generator = BatchGenerator(train_dataset,train_batch_size, N = crop)
    val_generator = BatchGenerator(val_dataset,val_batch_size, N = crop)
    lrate_sch = LearningRateScheduler(schedule)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')
    chkpt = ModelCheckpoint(path_train+'models/single_ptych/chkpt_'+model_name+'.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # callbacks_list = [early_stop]
    callbacks_list = [lrate_sch,early_stop, chkpt]

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
    adam = Adam(lr = lrate)

    # model.compile(loss='mean_squared_error',
    #           optimizer=sgd)

    model.compile(loss='mean_squared_error',
              optimizer=adam)

    # print validation_steps

    model.fit_generator(train_generator,validation_data=val_generator,
         validation_steps = validation_steps,steps_per_epoch = steps_per_epoch, 
         epochs = epochs,verbose=1 ,callbacks=callbacks_list)
    # model.fit(data, label, batch_size=train_batch_size, nb_epoch=epochs, verbose=1, callbacks=callbacks_list, validation_split=0.1, shuffle=True)
    model.save(path_train+'models/single_ptych/'+model_name+'.h5')

    # print model.summary()


def main():
    global lrate

    path_train =  "/home/sushobhan/Documents/data/fourier_ptychography/"
    home = "/home/sushobhan/Documents/research/fourier_ptychography/"
    # set_name = sys.argv[1]
    model_name = sys.argv[1]

    N = 64
    crop = 8
    lrate = 0.0001

    mParam = {}
    mParam['lrate'] = lrate
    mParam['epochs'] = int(sys.argv[2])
    mParam['decay'] = 0.0
    mParam['border_mode'] = 'valid'

    if mParam['border_mode'] == 'valid':
        mParam['crop'] = crop
    else:
        mParam['crop'] = 0

    mParam['input_shape'] = (1,None, None)
    mParam['output_shape'] = (1, None, None)

    mParam['train_batch_size'] = 16
    mParam['val_batch_size'] = 16
    mParam['steps_per_epoch'] = 12000//mParam['train_batch_size']
    mParam['validation_steps'] = 1500//mParam['val_batch_size']
    # mParam['set_name'] = set_name

    train_model(path_train,home,model_name,mParam)


if __name__ == '__main__':
    main()