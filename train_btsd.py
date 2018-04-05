import tensorflow as tf
import datetime
import os
# from yolo_models import Original_Yolo_Model
from btsd_yolo import BTSD_Yolo_Model
from timer import Timer
from pascal_voc import pascal_voc
import numpy as np
from architectures import *
from btsd import btsd_reader

slim = tf.contrib.slim

def log(message,file_path):

    f1=open(file_path, 'a+')
    f1.write(message)
    f1.close()

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def train(net,data):

       
        output_dir = os.path.join('saves', 'trained_initial')
        # output_dir = os.path.join(
        #     'saves', 'new BTSD'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # variable_to_restore = tf.global_variables()
        # variable_to_restore = slim.get_variables_to_restore(exclude=['fc33', 'fc34',\
        #  'dropout_35','fc_36'])
        # variable_to_restore = slim.get_variables_to_restore(exclude=['fc_33_btsd', 'fc_34_btsd',\
        # 'dropout_35_btsd','fc_36_btsd'])
        # print variable_to_restore
        # exit(0)
        # variable_to_save = slim.get_model_variables()
        # global_step =tf.get_variable('global_step', [], \
        # initializer=tf.constant_initializer(0), trainable=False)

        variable_to_save = tf.global_variables()
        # print variable_to_save
        # exit(0)
        restorer = tf.train.Saver(variable_to_save, max_to_keep=None)
        saver = tf.train.Saver(variable_to_save, max_to_keep=None)
        ckpt_file = os.path.join(output_dir, 'save.ckpt')
        weights_file = os.path.join('weights','YOLO_small.ckpt')
        log_path = os.path.join(output_dir,'log.txt')
        

        lr = net.learning_rate
        # 0.0001
        lr = tf.convert_to_tensor(lr, np.float32)
        decay_steps = 1000
        decay_rate = 0.1

        global_step =tf.get_variable('global_step', [], \
        initializer=tf.constant_initializer(0), trainable=False)


        learning_rate = tf.train.exponential_decay(
            lr, global_step, decay_steps,
            decay_rate, True, name='learning_rate')
        # print learning_rate
        # exit(0)
        sess = tf.Session()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "yolo/trainable/")

        # print len(train_vars)
        # all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print len(all_vars)
        # # print all_vars
        # initial_weights = list(set(all_vars)-set(train_vars))


        # print initial_weights
        # print len(initial_weights)
        # exit(0)  
        # print train_vars
        # exit(0)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            net.total_loss,var_list=train_vars, global_step=global_step)

        # ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        # averages_op = ema.apply(tf.trainable_variables())
        # # print averages_op
        # with tf.control_dependencies([optimizer]):
        #     train_op = tf.group(averages_op)
        train_op = optimizer
        # print train_op
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print tf.trainable_variables(scope='yolo')[0]
        # print tf.trainable_variables(scope='yolo')[0].name
        # exit(0)
        # for x in tf.trainable_variables(scope='yolo'):
        #     if 'weight' in x.name:
        #         y = tf.reduce_mean(tf.abs(x))
        #         tf.summary.scalar(x.name,y,family='visualize layer variables')

        summary_op = tf.summary.merge_all()
        print 'summary tensor made'
        writer = tf.summary.FileWriter(output_dir,flush_secs=60)

        # intitial_weights_restorer = tf.train.Saver(initial_weights, max_to_keep=None)
        # if weights_file is not None:
        #     print('Restoring weights from: ' + weights_file)
        #     intitial_weights_restorer.restore(sess,weights_file)

        # print 'restoring weights from original yolo : ',weights_file
        # tf.contrib.framework.assign_from_checkpoint_fn(weights_file,variable_to_restore)

        print 'restoring session from : ', ckpt_file+'-1201'
        # print restorer._var_list
        # exit(0)
        restorer.restore(sess,ckpt_file+'-1201')

        
        
        writer.add_graph(sess.graph)

        print 'started training ...'
        train_timer = Timer()
        load_timer = Timer()
        # sess.run(tf.global_variables_initializer())

        initialize_uninitialized(sess)
        print 'variables initialised'

        # print 'restoring weights ...'
        # restorer.restore(session, weights_file)

        print_itr = 25
        save_itr = 200
        log_itr = 5


        

        for step in xrange(0, 5000):

            load_timer.tic()
            images, labels = data.get()
            load_timer.toc()
            feed_dict = {net.images: images, net.labels: labels}
            # print step

            # try:
            if(step%print_itr==0):

                # print 'in print_itr'
                train_timer.tic()
                summary_str, loss, _,curr_lr = sess.run(
                    [summary_op, net.total_loss,train_op,learning_rate],
                    feed_dict=feed_dict)
                train_timer.toc()

                log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f} Speed: {:.3f}s/iter,'
                        ' Load: {:.3f}s/iter').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        data.epoch,
                        int(step),
                        learning_rate.eval(session=sess),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time)

                print log_str
                log(log_str,log_path)
                # exit(0)
                writer.add_summary(summary_str, step)
            
            elif(step%log_itr==0):

                # print 'in log_itr'
                train_timer.tic()
                summary_str, _ = sess.run(
                    [summary_op, train_op],
                    feed_dict=feed_dict)
                train_timer.toc()

                writer.add_summary(summary_str, step)

            else:

                # print 'in else'
                train_timer.tic()
                sess.run(train_op, feed_dict=feed_dict)
                train_timer.toc()
            
            if(step%save_itr==0):

                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    output_dir))
                saver.save(sess, ckpt_file,
                                global_step=global_step)

            # except Exception as e:

            #     # log('EXCEPTION '+e,log_path)
            #     print e



                






def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    yolo = BTSD_Yolo_Model(trial_model1)
    # pascal = pascal_voc('train','/home/vikram_mm/yolo_tensorflow/data/')

    reader = btsd_reader('train','data',yolo.batch_size)

    print('Start training ...')
    train(yolo,reader)
    print('Done training.')

if __name__ == '__main__':

    main()