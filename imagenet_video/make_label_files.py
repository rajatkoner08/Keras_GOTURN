import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import time
import random
import os
import sys

basedir = os.path.dirname(__file__)
print('base dir ',basedir)
sys.path.append(os.path.abspath(os.path.join(
    basedir,
    os.path.pardir,
    os.path.pardir,
    os.path.pardir)))


def get_image_size(fname):
    import struct, imghdr, re
    '''Determine the image type of fhandle and return its size.
    from draco'''
    # Only a loop so we can break. Should never run more than once.
    while True:
        with open(fname, 'rb') as fhandle:
            head = fhandle.read(32)
            if len(head) != 32:
                break
            if imghdr.what(fname) == 'png':
                check = struct.unpack('>i', head[4:8])[0]
                if check != 0x0d0a1a0a:
                    break
                width, height = struct.unpack('>ii', head[16:24])
            elif imghdr.what(fname) == 'gif':
                width, height = struct.unpack('<HH', head[6:10])
            elif imghdr.what(fname) == 'jpeg':
                try:
                    fhandle.seek(0) # Read 0xff next
                    size = 2
                    ftype = 0
                    while not 0xc0 <= ftype <= 0xcf:
                        fhandle.seek(size, 1)
                        byte = fhandle.read(1)
                        while ord(byte) == 0xff:
                            byte = fhandle.read(1)
                        ftype = ord(byte)
                        size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                    fhandle.seek(1, 1)  # Skip `precision' byte.
                    height, width = struct.unpack('>HH', fhandle.read(4))
                except Exception: #IGNORE:W0703
                    break
            elif imghdr.what(fname) == 'pgm':
                header, width, height, maxval = re.search(
                    b"(^P5\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", head).groups()
                width = int(width)
                height = int(height)
            elif imghdr.what(fname) == 'bmp':
                _, width, height, depth = re.search(
                    b"((\d+)\sx\s"
                    b"(\d+)\sx\s"
                    b"(\d+))", str).groups()
                width = int(width)
                height = int(height)
            else:
                break
            return width, height
    imShape = cv2.imread(fname).shape
    return imShape[1], imShape[0]

DEBUG = False


def main( label_type):
    wildcard = '/*/*/' if label_type == 'train' else '/*/'
    dataset_path = os.path.join(os.path.dirname(__file__), 'ILSVRC/')
    annotationPath = dataset_path + 'Annotations/'
    imagePath = dataset_path + 'Data/'

    if not DEBUG:
        if not os.path.exists(os.path.join('labels', label_type)):
            os.makedirs(os.path.join('labels', label_type))
        imageNameFile = open('labels/' + label_type + '/image_names.txt', 'w')
        #imageNameFile = open(dataset_path + '/ImageSets/VID/train_1.txt', 'w')

    videos = sorted(glob.glob(annotationPath + 'VID/' + label_type + wildcard))
    print ('video ',videos)

    bboxes = []
    imageNameList = []
    imNum = 0
    totalImages = len(glob.glob(annotationPath + 'VID/' + label_type + wildcard + '*.xml'))
    print ('totalImages', totalImages)
    classes = {
            'n01674464': 1,
            'n01662784': 2,
            'n02342885': 3,
            'n04468005': 4,
            'n02509815': 5,
            'n02084071': 6,
            'n01503061': 7,
            'n02324045': 8,
            'n02402425': 9,
            'n02834778': 10,
            'n02419796': 11,
            'n02374451': 12,
            'n04530566': 13,
            'n02118333': 14,
            'n02958343': 15,
            'n02510455': 16,
            'n03790512': 17,
            'n02391049': 18,
            'n02121808': 19,
            'n01726692': 20,
            'n02062744': 21,
            'n02503517': 22,
            'n02691156': 23,
            'n02129165': 24,
            'n02129604': 25,
            'n02355227': 26,
            'n02484322': 27,
            'n02411705': 28,
            'n02924116': 29,
            'n02131653': 30,
            }

    for vv,video in enumerate(videos):
        labels = sorted(glob.glob(video + '*.xml'))
        images = [label.replace('Annotations', 'Data').replace('xml', 'JPEG') for label in labels]
        trackColor = dict()
        for ii,imageName in enumerate(images):
            if imNum % 100 == 0:
                print ('imNum %d of %d = %.2f%%' % (imNum, totalImages, imNum * 100.0 / totalImages))
            if not DEBUG:
                # Leave off initial bit of path so we can just add parent dir to path later.
                imageNameFile.write(imageName + '\n')
                imageNameList.append(imageName)
            label = labels[ii]
            labelTree = ET.parse(label)
            imgSize = get_image_size(images[ii])
            area = imgSize[0] * imgSize[1]
            if DEBUG:
                print ('\n%s' % images[ii])
                image = cv2.imread(images[ii])
                print ('video', vv, 'image', ii)
            for obj in labelTree.findall('object'):
                cls = obj.find('name').text
                assert cls in classes
                classInd = classes[cls]

                occl = int(obj.find('occluded').text)
                trackId = int(obj.find('trackid').text)
                bbox = obj.find('bndbox')
                bbox = [images[ii], int(bbox.find('xmin').text),
                        int(bbox.find('ymin').text),
                        int(bbox.find('xmax').text),
                        int(bbox.find('ymax').text),
                        vv, trackId, imNum, classInd, occl]

                if DEBUG:
                    print ('name', obj.find('name').text, '\n')
                    print (bbox)
                    if trackId not in trackColor:
                        trackColor[trackId] = [random.random() * 255 for _ in range(3)]
                    #drawRect(image, bbox[:4], 3, trackColor[trackId])
                bboxes.append(bbox)
            #if DEBUG:
                #cv2.imshow('image', image)
                #cv2.waitKey(1)

            imNum += 1

    bboxes = np.array(bboxes)
    print("")
    # Reorder by video_id, then track_id, then video image number so all labels for a single track are next to each other.
    # This only matters if a single image could have multiple tracks.
    order = np.lexsort((bboxes[:,6], bboxes[:,5], bboxes[:,4]))
    bboxes = bboxes[order,:]
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)
    return bboxes, imageNameList, totalImages

if __name__ == '__main__':
    main('train')
#    main('val')

