import argparse
import os
import sys
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback, TensorBoard
from keras import backend as K
from keras.layers import Input,Dense,merge,Concatenate,Lambda,Merge,Flatten,Dropout,concatenate,regularizers
from keras.models import load_model,Model
import numpy as np
import tensorflow as tf

L2_WEIGHT_PENALTY = 0.0005
Epoch =1024
Batch_Size = 64
LearningRate = 0.0001

from DataGenerator import *
from alexnet import alex_net_conv_layers
from vgg16 import VGG16
#from vgg16_skip import VGG16

smooth = 1

#weight = '/weight/alexnet_weights.h5'
weight = 'weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
log_dir = os.path.join(basedir,'logs')

l2_loss = regularizers.l2(L2_WEIGHT_PENALTY)



# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch <= 10:
        return 0.0001
    elif epoch >10 and epoch <1000:
        return 0.00001
    else:
        return 0.000001

 ### IOU or dice coeff calculation
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def full_loss(outputs, labels):
    diff = K.sum(K.abs(outputs - labels), axis=1)
    loss = K.mean(diff)
    return loss

# 2: Build the Keras model (and possibly load some trained weights)
def main(FLAGS):

    debug = FLAGS.debug or FLAGS.output

    datagen = DataGenerator(batch_size=Batch_Size)
    print 'train data len : ',len(datagen.train_indices)
    print 'validation data len : ', len(datagen.val_indices)

    input_dim = [Batch_Size, 224, 224, 3]


    K.clear_session() # Clear previous models from memory.

    image_input = Input(batch_shape=input_dim)
    #feature_model = alex_net_conv_layers(image_input, weights_path= weight)
    #feature_model = vgg16_skip(input_tensor=image_input,weights_path=weight,pooling='max')
    feature_model = VGG16(image_input)

    trainable_layer = ['block1_skip', 'block2_skip','block3_skip','block4_skip',
                       'block1_prelu', 'block2_skip', 'block3_prelu', 'block4_prelu',
                       'block1_skip_flat', 'block2_skip_flat', 'block3_skip_flat','block4_skip_flat',
                       'large_concat']


    # for i, layer in enumerate(feature_model.layers):
    #     print(i, layer.name)


    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in feature_model.layers:
        if layer.name not in trainable_layer:
            layer.trainable = False

    img_t0 = Input(batch_shape=input_dim)
    img_t1 = Input(batch_shape=input_dim)
    # because we re-use the same instance of the network
    # the weights of the network will be shared across the two branches
    img_feature_t0 = feature_model(img_t0)
    img_feature_t1 = feature_model(img_t1)

    concat = Concatenate(axis=-1)
    merge_images = concat([img_feature_t0,img_feature_t1])
    print ' Merged images shape ',K.get_variable_shape(merge_images)

    # Embeded fully connected layer of 4096
    fc6 = Dense(4096, activation='relu',kernel_regularizer=l2_loss, name='fc6')(merge_images)
    fc6_dropout = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu',kernel_regularizer=l2_loss)(fc6_dropout)
    fc7_dropout = Dropout(0.5)(fc7)
    # fc8 = Dense(4096, activation='relu',kernel_regularizer=l2_loss)(fc7_dropout)
    # fc8_dropout = Dropout(0.5)(fc8)
    #predict bounding box of the target
    bbox_out = Dense(4,name='fc_bbox')(fc7_dropout)

    #https://sorenbouma.github.io/blog/oneshot/   --siamese one shot


    model = Model(input=[img_t0, img_t1], output=bbox_out)
    print 'Final model sumary ',model.summary()

    #model.load_weights(log_dir + '/re3_mobile_weights_epoch-08_loss-63.3082_val_loss-59.1573.h5',by_name=True)

    model.compile(loss=full_loss, optimizer=Adam(lr=LearningRate), metrics=['accuracy'])
    #model.compile(loss=full_loss, optimizer=SGD(lr=LearningRate, momentum=0.9), metrics=['accuracy'])

    # Generators
    training_generator = datagen.train_generate()
    validation_generator = datagen.val_generate()

    #[img0,img1],gt = next(training_generator)

    print'Starting training ....'

    #Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=10,
                                  epochs=Epoch,
                                  #initial_epoch=10,
                         callbacks=[#ModelCheckpoint(log_dir+'/re3_Alex_weights_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                        #               monitor='val_loss', verbose=2,
                        #               save_best_only=True,
                        #               save_weights_only=True,
                        #               mode='auto',
                        #               period=1),
                                            TensorBoard(log_dir='./logs',
                                                                          histogram_freq=0, write_graph=True, write_images=True)],
                                  validation_data=validation_generator,
                                  validation_steps= 1)

    model_name = 'goturn_vgg16'
    model.save(log_dir+'/{}.h5'.format(model_name))
    model.save_weights(log_dir+'/{}_weights.h5'.format(model_name))
    print()
    print("Model saved as {}.h5".format(model_name))
    print("Weights also saved separately as {}_weights.h5".format(model_name))
    print()

    # #make prediction
    # [img0, img1], y_list = next(predict_generator)
    #
    # # for i in range(img0.shape[0]):
    # #     datagen.showData(img0[i,:],img1[i,:],y_list[i,:], y_list[i,:])
    #
    # y_pred = model.predict([img0,img1])
    # #print 'oringinal gt ',y_list,' and predicted box ',y_pred
    # print ' printing predicted data'
    # for i in range(img0.shape[0]):
    #     datagen.showData(img1[i,:],img1[i,:],y_list[i,:], y_pred[i,:])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for GoTurn VGG16.')
    parser.add_argument('-r', '--restore', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-o', '--output', action='store_true', default=False)
    parser.add_argument('-c', '--clear_snapshots', action='store_true', default=False, dest='clearSnapshots')
    parser.add_argument('-p', '--port', action='store', default=9987, dest='port', type=int)
    parser.add_argument('--run_val', action='store_true', default=False)
    parser.add_argument('--val_device', type=str, default='0', help='Device number or string for val process to use.')
    parser.add_argument('-m', '--max_steps', type=int, default=10000, help='Number of steps to run trainer.')
    FLAGS = parser.parse_args()
    main(FLAGS)
