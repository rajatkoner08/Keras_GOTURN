from __future__ import division
import sys
import argparse
import random
import cv2
import numpy as np
import os
import sys
from PIL import Image,ImageDraw,ImageFont,ImageOps
import time

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

from training import get_datasets
from training.unrolled_solver import add_noise
from training.unrolled_solver import fix_bbox_intersection
from re3_utils.util import im_util
from re3_utils.util import bb_util
from re3_utils.util import drawing
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import GPU_Cluster

CROP_SIZE =224
NO_OF_CHANEL = 3
CROP_PAD = 2

USE_SIMULATER = 0.5
USE_NETWORK_PROB = 0.8
REAL_MOTION_PROB = 1.0 / 8
AREA_CUTOFF = 0.25
USE_MIRROR_PROB = 0.5

random_seed = 123


class DataGenerator(object):

    #Generates  data  for Keras
    def __init__(self,  dataset_type = 'train',batch_size=1, debug=True, shuffle = True):
        #'Initialization of set of constraint'
        self.debug = debug
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random.seed(random_seed)
        self.no_of_target = 1  #no of target present in the frame

        self.gt = []
        self.image_paths = []
        # all image number which has consicutive image as per unroll and same video track
        self.all_img_wd_gt = []
        self.__create_list_ids()

        #shuffle list
        self.indices = self.__get_exploration_order(self.all_img_wd_gt)
        #96% for training set, rest for validation
        self.train_indices = self.indices[:int(len(self.indices) * 0.96)]
        self.val_indices = self.indices[int(len(self.indices) * 0.96) + 1:]
        # Read in and format GT.
        # dict from (dataset_ind, video_id, track_id, image_id) to line in labels array
        self.key_lookup = dict()
        #self.__calculate_mean()
        self.img_no = 1



######################################################################################
    def train_generate(self):

        'Generates batches of samples for training'
        # Infinite loop
        while 1:
            # Generate batches for SOT per target, per unroll
            imax = len(self.train_indices) // self.batch_size
            for i in range(imax):
                X = []
                y = []

                list_IDs_temp = [self.all_img_wd_gt[k] for k in
                                 self.train_indices[i * self.batch_size:(i + 1) * self.batch_size]]
                # Find list of IDs on a batch)
                    #load batch images with ground truth a
                X , y = self.__batch_generate(list_IDs_temp)
                # output: img_t0, and img_t1,and gt of img_t1
                yield [X[:,0, :, :, :], X[:,1, :, :, :]], y


###########################################################################################
    def val_generate(self):
        'Generates batches of samples for validation'

        # Infinite loop
        while 1:
            # Generate batches for SOT per target, per unroll
            imax = len(self.val_indices) // self.batch_size
            for i in range(imax):
                X = []
                y = []
                list_IDs_temp = [self.all_img_wd_gt[k] for k in
                                 self.val_indices[i * self.batch_size:(i + 1) * self.batch_size]]
                #load batch images with ground truth as per list and digit number
                X , y = self.__batch_generate(list_IDs_temp)
                # output: img_t0, and img_t1,and gt of img_t1
                yield [X[:, 0, :, :, :], X[:, 1, :, :, :]], y

###########################################################################################

    def __batch_generate(self, list_IDs_temp):
        #Generates data of batch size samples
        # X : (n_samples, num_unrolls, row, col, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 2, CROP_SIZE, CROP_SIZE, NO_OF_CHANEL))
        y = np.empty((self.batch_size, 4))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,:,:,:,:],y[i,:] = self.__load_image(ID[0], ID[1])

        return X, y

###########################################################################################
    def __load_image(self, img_index, gt_index):

        #use mirror,noisy image,
        mirrored = random.random() < USE_MIRROR_PROB
        gtType = random.random()
        realMotion = gtType < REAL_MOTION_PROB

        #load images and gt as per number of unroll
        X = np.empty((2, CROP_SIZE, CROP_SIZE,NO_OF_CHANEL), dtype=np.float32)

        try:
            img_ind = int(img_index) #- 1  # as image start with 1 and list start with 0
            imagePath0 = self.image_paths[0][img_ind]
            imagePath1 = self.image_paths[0][img_ind + 1]

            # if GPU_Cluster:
            #     imagePath0= '/mnt/glusterdata/home/koner/'+imagePath0
            #     imagePath1 = '/mnt/glusterdata/home/koner/'+imagePath1

            if self.debug:
                if GPU_Cluster:
                    print 'Reading image', imagePath0[135:], ' and ', imagePath1[135:]
                else:
                    print 'Reading image', imagePath0[126:], ' and ', imagePath1[126:]
                print 'respective GT',self.gt[gt_index],' and ',self.gt[gt_index+1]
            #take initial box from starting image of unroll
            if GPU_Cluster:
                previous_box = self.gt[gt_index][1:5]
                # read next/curr frame gt
                curr_bbox = self.gt[(gt_index + 1)][1:5]
            else:
                previous_box = self.gt[gt_index][:4]
                curr_bbox = self.gt[(gt_index + 1)][:4]

            #read two consicutive images, with their common initial gt
            image_0 = cv2.imread(imagePath0)[:,:,::-1].astype(np.float32)
            # We normalize the colors (in RGB space) with the empirical means on the training set
            image_0[:, :, 0] -= 123.68
            image_0[:, :, 1] -= 116.779
            image_0[:, :, 2] -= 103.939
            image_1 = cv2.imread(imagePath1)[:, :, ::-1].astype(np.float32)
            image_1[:, :, 0] -= 123.68
            image_1[:, :, 1] -= 116.779
            image_1[:, :, 2] -= 103.939

            # add all the noise flipping,and corping of images
            if not realMotion :
                noisy_box = add_noise(previous_box.astype('float32').tolist(), previous_box.astype('float32').tolist(), image_0.shape[1], image_0.shape[0])
            else:
                noisy_box = fix_bbox_intersection(previous_box.astype('float32'), curr_bbox.astype('float32'), image_0.shape[1], image_0.shape[0])

            X[0, ...] = im_util.get_cropped_input(image_0, previous_box.astype('float32'), CROP_PAD, CROP_SIZE)[0]

            X[1, ...] = im_util.get_cropped_input(image_1, noisy_box.astype('float32'), CROP_PAD, CROP_SIZE)[0]

            shiftedBBox = bb_util.to_crop_coordinate_system( curr_bbox,noisy_box, CROP_PAD, 1)
            shiftedBBoxXYWH = bb_util.xyxy_to_xywh(shiftedBBox)

            if mirrored :
                X = np.fliplr(
                    X.transpose(1, 2, 3, 0)).transpose(3, 0, 1, 2)
                shiftedBBoxXYWH = np.array([1-shiftedBBoxXYWH[0], shiftedBBoxXYWH[1],shiftedBBoxXYWH[2],shiftedBBoxXYWH[3]])

            if self.debug:
                self.showData(image_0,image_1,X[0,...], X[1,...], shiftedBBoxXYWH )

            xyxyLabels = bb_util.xywh_to_xyxy(shiftedBBoxXYWH)
            xyxyLabels = xyxyLabels*10



        except Exception as ex:
            import traceback
            trace = traceback.format_exc()
            print trace
            errorFile = open('error.txt', 'a+')
            errorFile.write('exception in lookup_func %s\n' % str(ex))
            errorFile.write(str(trace))

        return X,xyxyLabels


###########################################################################################
    def __get_exploration_order(self, all_keys):
        'randomly shuffle image order if shufle is true'
        # Find exploration order
        indexes = np.arange(len(all_keys))
        if self.shuffle == True:
            random.Random(random_seed).shuffle(indexes)

        return indexes
############################################################################################
    #todo calculate each chanel mean
    def __calculate_mean(self):
        avg_mean = 0
        for i, ID in enumerate(self.train_indices):
            ind = int(ID) - 1  # list start with inex 0, and image 1
            path = self.image_paths[0][ind]
            image = Image.open(path)
            image = np.array(image)
            avg_mean += np.mean(image)
            #print 'sum ',sum
        avg_mean = avg_mean/(image.shape[0]*image.shape[1])
        print 'avg mean ',avg_mean
        exit()
        self.mean = avg_mean


############################################################################################

    def __create_list_ids(self):
        datasetName = 'imagenet_video'
        #datasetName = 'MNIST_train'
        #no of object present in the frame
        no_of_object = 1
        self.__add_dataset(dataset_name=datasetName,no_of_object=no_of_object)


    def __add_dataset(self, dataset_name,no_of_object):
        dataset_ind = len(self.image_paths)
        data = get_datasets.get_data_for_dataset(dataset_name, 'train')
        self.gt = data['gt']
        num_keys = 0
        stop = len(self.gt) - 1  #two consicutive images
        for xx in xrange(0, stop):
            start_line = self.gt[xx,:]
            end_line = self.gt[xx + 1,:]
            # Check that still in the same sequence.
            # Video_id should match, track_id should match, and image number should be exactly num_unrolls frames later.
            if (start_line[4] == end_line[4] and
                start_line[5] == end_line[5] and
                int(start_line[6]) + 1 == int(end_line[6])):
                # Add image id and respective ground truth.
                self.all_img_wd_gt.append([str(start_line[6]).rjust(6, '0'), xx])
                num_keys += 1
            if self.debug:
                print '#%s added image : %s' % (dataset_name, start_line[6])

        image_paths = data['image_paths']
        # Add the array to image_paths. Note that image paths is indexed by the dataset number THEN by the image line.
        self.image_paths.append(image_paths)

###################################################################################
    def showData(self,big_img0,big_img1, img_t0, img_t1, gt_rect1):
        # if not os.path.exists(basedir+'/debugData'):
        #     os.mkdir(basedir+'/debugData')
        # else:
        #for i in xrange(len(img_t0)):
        # Look at the inputs to make sure they are correct.


        image0 = img_t0.copy()
        image1 = img_t1.copy()

        xyxyLabel = bb_util.xywh_to_xyxy((gt_rect1).squeeze())
        print 'xyxy raw', xyxyLabel, 'actual', xyxyLabel * CROP_PAD
        label = np.zeros((CROP_PAD, CROP_PAD))
        drawing.drawRect(label, xyxyLabel * CROP_PAD, 0, 1)
        drawing.drawRect(image0, bb_util.xywh_to_xyxy(np.full((4, 1), .5) * CROP_SIZE), 2, [255, 0, 0])
        drawing.drawRect(image1, xyxyLabel * CROP_SIZE, 1, [0, 255, 0])
        bigImage0 = big_img0.copy()
        bigImage1 = big_img1.copy()
        # if dd < len(cropBBoxes):
        #     drawing.drawRect(bigImage1, bboxes[dd], 5, [255,0,0])
        #     drawing.drawRect(image1, cropBBoxes[dd] * CROP_SIZE, 1, [0,255,0])
        #     print 'pred raw', cropBBoxes[dd], 'actual', cropBBoxes[dd] * CROP_PAD
        print '\n'

        label[0, 0] = 1
        label[0, 1] = 0
        plots = [bigImage0, bigImage1, image0, image1, label]
        subplot = drawing.subplot(plots, 3, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5)
        cv2.imwrite('./debugedImage/'+str(self.img_no).rjust(6, '0')+'.jpg',subplot[:, :, ::-1])
        self.img_no = self.img_no + 1
        if(self.img_no == 256 ):
            exit()
        # cv2.imshow('debug', subplot[:, :, ::-1])
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()


#####################################################################################
def main(args):
    datagen = DataGenerator(args)
    X,Z,y = zip(*datagen.train_generate())
    #X, Z, y = zip(*datagen.val_generate())
    print 'value of y : ',y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for network images.')

    parser.add_argument('-n', '--num_unrolls', action='store', default=4,
            dest='num_unrolls', type=int)

    parser.add_argument('-d', '--debug', action='store_true', default=True,
            dest='debug')
    parser.add_argument('-b', '--batch', action='store', default=1,
                        dest='batch_size',type=int)
    args = parser.parse_args()
    main(args)


