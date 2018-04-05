import numpy as np
import tensorflow as tf


slim = tf.contrib.slim



class original_yolo_net():

    def __init__(self,input,alpha,num_outputs,is_training,yolo_cell_size=7):
        
        with tf.variable_scope('yolo'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], #using scope to avoid mentioning the paramters repeatdely
                                        activation_fn=self.lrelu(alpha),
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        net = tf.pad(input, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                        net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                    
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                    
                        net = slim.conv2d(net, 192, 3, scope='conv_4')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                        net = slim.conv2d(net, 128, 1, scope='conv_6')
                        net = slim.conv2d(net, 256, 3, scope='conv_7')
                        net = slim.conv2d(net, 256, 1, scope='conv_8')
                        net = slim.conv2d(net, 512, 3, scope='conv_9')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                        net = slim.conv2d(net, 256, 1, scope='conv_11')
                        net = slim.conv2d(net, 512, 3, scope='conv_12')
                        net = slim.conv2d(net, 256, 1, scope='conv_13')
                        net = slim.conv2d(net, 512, 3, scope='conv_14')
                        net = slim.conv2d(net, 256, 1, scope='conv_15')
                        net = slim.conv2d(net, 512, 3, scope='conv_16')
                        net = slim.conv2d(net, 256, 1, scope='conv_17')
                        net = slim.conv2d(net, 512, 3, scope='conv_18')
                        net = slim.conv2d(net, 512, 1, scope='conv_19')
                        net = slim.conv2d(net, 1024, 3, scope='conv_20')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                        net = slim.conv2d(net, 512, 1, scope='conv_22')
                        net = slim.conv2d(net, 1024, 3, scope='conv_23')
                        net = slim.conv2d(net, 512, 1, scope='conv_24')
                        net = slim.conv2d(net, 1024, 3, scope='conv_25')
                        net = slim.conv2d(net, 1024, 3, scope='conv_26')
                        net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                        net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                        net = slim.conv2d(net, 1024, 3, scope='conv_29')
                        net = slim.conv2d(net, 1024, 3, scope='conv_30')
                        net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                        net = slim.flatten(net, scope='flat_32')
                        net = slim.fully_connected(net, 512, scope='fc_33')
                        net = slim.fully_connected(net, 4096, scope='fc_34')
                        net = slim.dropout(net,is_training=is_training, scope='dropout_35')
                        # net = slim.fully_connected(net, num_outputs=(yolo_cell_size*yolo_cell_size)*(20+2*5), #20 - num_classes,
                                                                                                            #2 - boxes per cell
                                                                                                            #5 - x,y,h,w,confidence
                                                # activation_fn=None, scope='fc_36')
                        
                        net = slim.fully_connected(net, num_outputs,
                                            activation_fn=None, scope='fc_36')
            self.net = net

    def lrelu(self,alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op


class trial_model1():

    def __init__(self,input,alpha,num_outputs,is_training,yolo_cell_size=7):
        
        with tf.variable_scope('yolo'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], #using scope to avoid mentioning the paramters repeatdely
                                        activation_fn=self.lrelu(alpha),
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        net = tf.pad(input, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                        net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                    
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                    
                        net = slim.conv2d(net, 192, 3, scope='conv_4')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                        net = slim.conv2d(net, 128, 1, scope='conv_6')
                        net = slim.conv2d(net, 256, 3, scope='conv_7')
                        net = slim.conv2d(net, 256, 1, scope='conv_8')
                        net = slim.conv2d(net, 512, 3, scope='conv_9')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                        net = slim.conv2d(net, 256, 1, scope='conv_11')
                        net = slim.conv2d(net, 512, 3, scope='conv_12')
                        net = slim.conv2d(net, 256, 1, scope='conv_13')
                        net = slim.conv2d(net, 512, 3, scope='conv_14')
                        net = slim.conv2d(net, 256, 1, scope='conv_15')
                        net = slim.conv2d(net, 512, 3, scope='conv_16')
                        net = slim.conv2d(net, 256, 1, scope='conv_17')
                        net = slim.conv2d(net, 512, 3, scope='conv_18')
                        net = slim.conv2d(net, 512, 1, scope='conv_19')
                        net = slim.conv2d(net, 1024, 3, scope='conv_20')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                        net = slim.conv2d(net, 512, 1, scope='conv_22')
                        net = slim.conv2d(net, 1024, 3, scope='conv_23')
                        net = slim.conv2d(net, 512, 1, scope='conv_24')
                        net = slim.conv2d(net, 1024, 3, scope='conv_25')
                        net = slim.conv2d(net, 1024, 3, scope='trainable/conv_26')
                        net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='trainable/pad_27')
                        net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='trainable/conv_28')
                        net = slim.conv2d(net, 1024, 3, scope='trainable/conv_29')
                        net = slim.conv2d(net, 1024, 3, scope='trainable/conv_30')
                        net = tf.transpose(net, [0, 3, 1, 2], name='trainable/trans_31')
                        net = slim.flatten(net, scope='trainable/flat_32')
                        net = slim.fully_connected(net, 512, scope='trainable/fc_33_btsd')
                        net = slim.fully_connected(net, 4096, scope='trainable/fc_34_btsd')
                        net = slim.dropout(net,is_training=is_training, scope='trainable/dropout_35_btsd')
                        # net = slim.fully_connected(net, num_outputs=(yolo_cell_size*yolo_cell_size)*(20+2*5), #20 - num_classes,
                                                                                                            #2 - boxes per cell
                                                                                                            #5 - x,y,h,w,confidence
                                                # activation_fn=None, scope='fc_36')
                        
                        net = slim.fully_connected(net, num_outputs,
                                            activation_fn=None, scope='trainable/fc_36_btsd')
            self.net = net

    def lrelu(self,alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op

	
if __name__ == '__main__':

    inp = tf.placeholder(tf.float32, shape=(None,448,448,3))
    obj = original_yolo_net(inp,0.004,True)
    print obj.net
