from keras.layers import MaxPooling2D,ZeroPadding2D,Dropout
from keras.layers import Conv2D,Dense,BatchNormalization,Flatten,InputLayer,Input,merge,Activation
from keras.models import Model

from keras.layers.advanced_activations import PReLU
from keras.initializers import constant
from keras import backend as K
import keras

from convonetutil import *

p_int = constant(0.25)


def alex_net_conv_layers(img_input, weights_path=None, heatmap=False, toplayer = False):
    if heatmap:
        inputs = Input(shape=(3, None, None))
    else:
        inputs = img_input

    conv_1 = Conv2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = BatchNormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Conv2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
        Conv2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
        Conv2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Conv2D(4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Conv2D(4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Conv2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    elif toplayer:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, name='dense_3')(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)
    else:
        prediction = dense_1

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model
