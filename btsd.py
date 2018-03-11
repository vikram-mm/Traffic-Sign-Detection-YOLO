import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy



class btsd_reader(object):
    def __init__(self, phase,data_path,batch_size=45,rebuild=False,flipped=True):

        self.data_path = os.path.join(data_path, 'btsd')
        self.cache_path = os.path.join(data_path, 'cache')
        self.batch_size = batch_size
        self.image_size = 448
        self.cell_size = 7
        # self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = flipped
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 18))
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
        image = (image / 255.0) #* 2.0 - 1.0
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
        cache_file = os.path.join(self.cache_path, 'btsd_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = cPickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'annotation',
                                   'training.txt')
        else:
            txtname = os.path.join(self.data_path, 'annotation',
                                   'testing.txt')
        
        with open(txtname, 'r') as f:
            self.image_index = list(set(x.split(';')[0] for x in f.readlines()))
        
        # print self.image_index
        # print len(self.image_index)
        # print len(self.create_dict(txtname))
        # exit(0)
        self.mapping = self.create_dict(txtname)
        gt_labels = []
        for i,index in enumerate(self.image_index):
            if(i%100==0):
                print i,index
            label, num = self.load_btsd_annotation(i,index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path,index)
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_labels, f)
        return gt_labels

    def create_dict(self,txtfile):

        f = open(txtfile)
        mapping = {}
        for line in f.readlines():

            a = line.split(';')

            if a[0] in mapping:
                mapping[a[0]].append((a[1],a[2],a[3],a[4],a[5],a[6]))
            else:
                mapping[a[0]] = [(a[1],a[2],a[3],a[4],a[5],a[6])]
        f.close()
        return mapping





    def load_btsd_annotation(self,i,index):

        
        imname = os.path.join(self.data_path,index)
        # print imname
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 18))
       
        objs = self.mapping[index]

        for obj in objs:
            
            # Make pixel indexes 0-based
            x1 = max(min((float(obj[0]) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(obj[1]) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(obj[2]) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(obj[3]) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = int(obj[5])
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # print 'b ->',boxes
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1
        
        #     print label[y_ind,x_ind]
        # print len(objs)
        # exit(0)
        if(i%100==0):
            all_boxes = []
            for obj in objs:
                x1 = max(min((float(obj[0]) - 1) * w_ratio, self.image_size - 1), 0)
                y1 = max(min((float(obj[1]) - 1) * h_ratio, self.image_size - 1), 0)
                x2 = max(min((float(obj[2]) - 1) * w_ratio, self.image_size - 1), 0)
                y2 = max(min((float(obj[3]) - 1) * h_ratio, self.image_size - 1), 0)
                cls_ind = int(obj[5])
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                all_boxes.append(boxes)

            self.draw_result(im,all_boxes,os.path.join('vis',str(i)+'.jpg'))
        return label, len(objs)

    def draw_result(self,img,result,path):

            # cv2.imwrite('original'+path,img)
            # print len(result)
            img = cv2.resize(img, (self.image_size, self.image_size))
            for i in range(len(result)):
                x = int(result[i][0])
                y = int(result[i][1])
                w = int(result[i][2] / 2)
                h = int(result[i][3] / 2)
                cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                
            
            
            cv2.imwrite(path,img)
        
if __name__ == '__main__':

    btsd = btsd('train','data')
