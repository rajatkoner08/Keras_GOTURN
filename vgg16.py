# -*- coding: utf-8 -*-
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import print_function
from __future__ import absolute_import
import os

import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from  keras.applications.imagenet_utils import decode_predictions
from  keras.applications.imagenet_utils import preprocess_input
from  keras.applications.imagenet_utils import _obtain_input_shape
import keras

basedir = os.path.dirname(__file__)
weight_path = os.path.join(basedir,'weight')
#WEIGHTS_PATH_NO_TOP = 'weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

from keras.layers.advanced_activations import PReLU
from keras.initializers import constant
p_int = constant(0.25)

pooling='Max'
classes=1000
include_top=False

def VGG16(img_input):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block1 skip
    # block1_skip = Conv2D(8, kernel_size=1, strides=1, name='block1_skip')(x)
    # block1_skip = PReLU(alpha_initializer=p_int, name='block1_prelu')(block1_skip)
    # block1_skip = Flatten(name='block1_skip_flat')(block1_skip)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 2 skip
    block2_skip = Conv2D(16, kernel_size=1, strides=1, name='block2_skip')(x)
    block2_skip = PReLU(alpha_initializer=p_int, name='block2_prelu')(block2_skip)
    block2_skip = MaxPooling2D((2,2))(block2_skip)
    block2_skip = Flatten(name='block2_skip_flat')(block2_skip)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #block 3 skip connection
    block3_skip = Conv2D(16, kernel_size=1, strides=1, name='block3_skip')(x)
    block3_skip = PReLU(alpha_initializer=p_int, name='block3_prelu')(block3_skip)
    block3_skip = MaxPooling2D((2,2))(block3_skip)
    block3_skip = Flatten(name='block3_skip_flat')(block3_skip)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #block 4 skip connection
    block4_skip = Conv2D(32, kernel_size=1, strides=1, name='block4_skip')(x)
    block4_skip = PReLU(alpha_initializer=p_int, name='block4_prelu')(block4_skip)
    block4_skip = MaxPooling2D((2,2))(block4_skip)
    block4_skip = Flatten(name='block4_skip_flat')(block4_skip)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    # block 5 skip connection
    block5_skip = Conv2D(32, kernel_size=1, strides=1, name='block5_skip')(x)
    block5_skip = PReLU(alpha_initializer=p_int, name='block5_prelu')(block5_skip)
    block5_skip = MaxPooling2D((2, 2))(block5_skip)
    block5_skip = Flatten(name='block5_skip_flat')(block5_skip)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if K.get_variable_shape(x).__len__() == 2:
        flat_x = x
    else:
        flat_x =  Flatten(name='vgg16_flat')(x)

    # concatinate all the layer
    concatenated = keras.layers.concatenate([ block2_skip,block3_skip, block4_skip, block5_skip, flat_x], name='large_concat')

    # Create model.
    model = Model(inputs=img_input, outputs=concatenated, name='vgg16')

    # # load weights
    if weight_path is not None:
        model.load_weights(weight_path+'/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    # print('conv block output 1 ',model.get_layer('block1_pool').output)
    # model.layers[3].set_weights()
    print('model summary ', model.summary())

    # skip_concat = keras.layers.concatenate([block2_skip, block3_skip, block4_skip, block5_skip], name='skip_concat')
    #
    # #stack tensor
    # K.stack([model,skip_concat])

    return model
