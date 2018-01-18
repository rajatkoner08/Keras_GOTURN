from keras.layers import MaxPooling2D,ZeroPadding2D,Dropout
from keras.layers import Conv2D,Dense,BatchNormalization,Flatten,InputLayer,Input,merge,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs

from keras.layers.advanced_activations import PReLU
from keras.initializers import constant
from keras import backend as K
import keras

from convonetutil import *

p_int = constant(0.25)


def vgg16_skip(input_tensor=None, weights_path=None, pooling=None):
    input_shape = [224,224,3]
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(img_input)

    block1_skip = Conv2D(8, kernel_size=1, strides=1)(x)
    block1_skip = PReLU(alpha_initializer=p_int)(block1_skip)
    block1_skip = Flatten()(block1_skip)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    block2_skip = Conv2D(16, kernel_size=1, strides=1)(x)
    block2_skip = PReLU(alpha_initializer=p_int)(block2_skip)
    block2_skip = Flatten()(block2_skip)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    block3_skip = Conv2D(32, kernel_size=1, strides=1)(x)
    block3_skip = PReLU(alpha_initializer=p_int)(block3_skip)
    block3_skip = Flatten()(block3_skip)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    block4_skip = Conv2D(64, kernel_size=1, strides=1)(x)
    block4_skip = PReLU(alpha_initializer=p_int)(block4_skip)
    block4_skip = Flatten()(block4_skip)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)


    #concatinate all the layer
    #concatenated = keras.layers.concatenate([block1_skip, block2_skip, block3_skip, block4_skip])

    # Create model.
    model = Model(img_input, x, name='vgg16_skip')

    # load weights
    if weights_path is not None:
        model.load_weights(weights_path,by_name=True)


    return model#,concatenated
