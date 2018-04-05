import tensorflow as tf
import datetime
import os
import argparse
from yolo_models import Original_Yolo_Model
from timer import Timer
from pascal_test_reader import pascal_voc
import skvideo.io
import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy



class pascal_voc(object):
    def __init__(self, phase,data_path,batch_size=45,rebuild=False,flipped=True):
        data_path = os.path.join(data_path,'pascal_voc')
        self.devkil_path = os.path.join(data_path, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.cache_path = os.path.join(data_path, 'cache')
        self.batch_size = batch_size
        self.image_size = 448
        self.cell_size = 7
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = flipped
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = cPickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main',
                                   'trainval.txt')
        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main',
                                   'val.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        # print imname
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # print 'b ->',boxes
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1
        
            # print label[y_ind,x_ind]
        # print len(objs)
        # exit(0)
        return label, len(objs)

if __name__ == '__main__':

    pascal = pascal_voc('train','/home/vikram_mm/yolo_tensorflow/data/')

sys.path.append('models/research/')
from models.research.object_detection.utils.object_detection_evaluation import *
# from utils.object_detection_evaluation import *
# from models.
# from object_detection import *
from object_detection.utils.object_detection_evaluation import *
# from object_detection_evaluation import *

from object_detection.core import standard_fields
# from object_detection.utils import object_detection_evaluation


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
        self.num_class = len(self.classes)
        self.image_size = 448
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def detect(self, img):

        # print 'detect'
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

       

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        # print 'dfcvm'
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        # print 'intre_out'
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        return boxes_filtered, probs_filtered, classes_num_filtered

        # result = []
        # for i in range(len(boxes_filtered)):
        #     result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
        #                   i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
                    
        
        
        # self.r = result
        # return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            # self.draw_result(frame, result)
            # cv2.imshow('Camera', frame)
            # cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        print 'idetect'
        detect_timer = Timer()
        image = cv2.imread(imname)
        # image = frame
        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        # self.draw_result(image, result)
        # cv2.imshow('Image', image)
        # cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    yolo = Original_Yolo_Model(False)
    # print yolo
    # exit(0)
    weight_file = os.path.join('weights','YOLO_small.ckpt')
    # weight_file = os.path.join('/home/vikram_mm/yolo_tensorflow/data/weights/YOLO_small.ckpt')
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    print "HERE"
    # videogen = skvideo.io.vreader('/home/vikram_mm/yolo_tensorflow/test/videoplayback.mp4')
    # for frame in videogen:
    #     print(frame.shape)
    #     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     detector.image_detector(frame, wait=40)
    #     #cv2.imshow('frame',frame)

       
    # # cv2.destroyAllWindows()
    # # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     print frame.shape
    # for k in ['000138.jpg','000147.jpg','000164.jpg'] :
    # imname = '000138.jpg'
    # imname = '000147.jpg'

    imname = 'person.jpg'
    detector.image_detector(imname)
    print detector.r

    # cap.release()
    cv2.destroyAllWindows()

def test(net,data):

    weight_file = os.path.join('weights','YOLO_small.ckpt')
    classes = CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'] 
    dict_list=[]
    i=1
    for x in classes:

        temp_dict={}
        temp_dict['id'] = i
        temp_dict['name'] = x
        dict_list.append(temp_dict)
        i+=1

    # evaluator = ObjectDetectionEvaluator(dict_list,0.5)
    evaluator =  PascalDetectionEvaluator(dict_list,0.5)
   
    print 'evaluator instantiated'
    detector = Detector(net, weight_file)

    eval_counter = 0
    # exit(0)
    for step in xrange(0, 109):

        # load_timer.tic()
        images, labels = data.get()
       
        feed_dict = {net.images: images, net.labels: labels}

        for i,image in enumerate(images):

            # print labels.shape
            object_present_indices = labels[i,:,:,0]==1

            # print object_present_indices

            gtbs= labels[i,object_present_indices,1:5]

            # print gtbs

            
            mgtb_list = [gtbs[:,1]-gtbs[:,3]/2,gtbs[:,0]-gtbs[:,2]/2,gtbs[:,1]+\
            gtbs[:,3]/2,gtbs[:,0]+gtbs[:,2]/2]
            # print mgtb_list
            mgtb = np.array(mgtb_list).transpose(1,0)
            # for x in mgtb_list:
            #     x = np.expand_dims(x,1)
            # # mgtb = np.array([gtbs[:,1]-gtbs[:,3]/2,gtbs[:,0]-gtbs[:,2]/2,gtbs[:,1]+\
            # # gtbs[:,3]/2,gtbs[:,0]+gtbs[:,2]/2])
            # mgtb = np.hstack(mgtb_list)
            # print mgtb
            # exit(0)
            class_ohlabels = labels[i,object_present_indices,5:]
            # print 'class_ohlabels \n' , class_ohlabels
            class_labels = np.where(class_ohlabels == 1)
            # print
            # print 'gt labels',class_labels[1]+1
            # print gtb
            # print gtb.shape
            # exit(0)

            evaluator.add_single_ground_truth_image_info(eval_counter,
            {standard_fields.InputDataFields.groundtruth_boxes: mgtb,
            standard_fields.InputDataFields.groundtruth_classes:
            class_labels[1]+1})
            # standard_fields.InputDataFields.groundtruth_difficult:
            # np.array([], dtype=bool)})

            # exit(0)

            # print'added ground_truth'

            # gt_dict[gt_counter] = {}

        # for i,image in enumerate(images):

            coords,probs,cl = detector.detect_from_cvmat(np.expand_dims(image,0))[0]
            # print 'pred labels ',cl+1
            # print 'pred coords ',coords
            # exit(0)
            coords_list = [coords[:,1]-coords[:,3]/2,coords[:,0]-coords[:,2]/2,coords[:,1]+\
            coords[:,3]/2,coords[:,0]+coords[:,2]/2]
            mcoords = np.array(coords_list).transpose(1,0)

            # print coords
            # print mcoords
            # boxes = detector.interpret_output(net_output)
            evaluator.add_single_detected_image_info(eval_counter,
            {standard_fields.DetectionResultFields.detection_boxes: mcoords,
            standard_fields.DetectionResultFields.detection_scores:
            probs,
            standard_fields.DetectionResultFields.detection_classes:
            cl+1})

            # print'added detection'

            eval_counter+=1

            if(eval_counter%100==0):
                print eval_counter
            # exit(0)
            # if(eval_counter==10):
            #     break

    print evaluator.evaluate()
        


def new_main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    yolo = Original_Yolo_Model()
    pascal = pascal_voc('test','/home/vikram_mm/yolo_tensorflow/data/')
    test(yolo,pascal)

    

#     print('Started testing ...')
#     train(yolo,pascal)
#     print('Done testing.')

if __name__ == '__main__':
    # main()
    new_main()
