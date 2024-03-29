import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

input_shape = (3, 256, 256)

# Create the model
model = Sequential()

model.add(Convolution2D(32, 9, 9, border_mode='same', input_shape=input_shape, init = 'he_normal'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=input_shape, init = 'he_normal'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=input_shape, init = 'he_normal'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(output_shape,init = 'he_uniform'))
model.add(BatchNormalization())
# model.add(Activation('sigmoid'))



# Compile model
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
model.save('model_CIFAR10_3C_epoch_20.h5')