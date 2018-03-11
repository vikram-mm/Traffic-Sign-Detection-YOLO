import numpy as np
import tensorflow as tf
from architectures import original_yolo_net

class Original_Yolo_Model():

    def __init__(self, is_training=True):

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
        self.num_class = len(self.classes)
        self.image_size = 448
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.object_scale = 1.0
        self.noobject_scale = 1.0
        self.class_scale = 2.0
        self.coord_scale = 5.0
        self.learning_rate = 0.0001
        self.batch_size = 45
        self.alpha = 0.1

        self.num_outputs = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell


        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

	#print self.offset
	#print self.offset.shape
	#exit(0)

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = original_yolo_net(self.images,self.alpha,self.num_outputs,is_training).net
        print self.logits

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    

    def loss_layer(self, predicts, labels, scope='loss_layer'):

        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1]) #1 if object is there
                                                                                                            #0 otherwise
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
            # print iou_predict_truth
            # exit(0)
            
            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response #why >= ?
            # print object_mask
            # exit(0)
            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss - if object is present - penalize predict_scales depending on how far from iou
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss - if object not present - penalise predict_scales just by it magnitude - ideally should be 0
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def calc_iou(self, boxes1, boxes2, scope='iou'):

	
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])

            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)



if __name__ == '__main__':

    Original_Yolo_Model(True)