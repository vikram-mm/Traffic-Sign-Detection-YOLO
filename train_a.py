import tensorflow as tf
import datetime
import os
from yolo_models import Original_Yolo_Model
from timer import Timer
from pascal_voc import pascal_voc
import numpy as np

slim = tf.contrib.slim

def log(message,file_path):

    f1=open(file_path, 'a+')
    f1.write(message)
    f1.close()


def train(net,data):

       

        output_dir = os.path.join(
            'saves', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # variable_to_restore = tf.global_variables()
        variable_to_restore = slim.get_model_variables()
        restorer = tf.train.Saver(variable_to_restore, max_to_keep=None)
        saver = tf.train.Saver(variable_to_restore, max_to_keep=None)
        ckpt_file = os.path.join(output_dir, 'save.ckpt')
        weights_file = os.path.join('weights','YOLO_small.ckpt')
        log_path = os.path.join(output_dir,'log.txt')
        

        lr = net.learning_rate
        lr = tf.convert_to_tensor(lr, np.float32)
        decay_steps = 30000
        decay_rate = 0.1

        global_step =tf.get_variable('global_step', [], \
        initializer=tf.constant_initializer(0), trainable=False)


        learning_rate = tf.train.exponential_decay(
            lr, global_step, decay_steps,
            decay_rate, True, name='learning_rate')
        # print learning_rate
        # exit(0)
        sess = tf.Session()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
            net.total_loss, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        averages_op = ema.apply(tf.trainable_variables())
        # print averages_op
        with tf.control_dependencies([optimizer]):
            train_op = tf.group(averages_op)
        # print train_op
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        summary_op = tf.summary.merge_all()
        print 'summary tensor made'
        writer = tf.summary.FileWriter(output_dir, flush_secs=60)

        if weights_file is not None:
            print('Restoring weights from: ' + weights_file)
            restorer.restore(sess,weights_file)
        
        writer.add_graph(sess.graph)

        print 'started training ...'
        train_timer = Timer()
        load_timer = Timer()
        sess.run(tf.global_variables_initializer())

        # print 'restoring weights ...'
        # restorer.restore(session, weights_file)

        print_itr = 25
        save_itr = 250
        log_itr = 5


        

        for step in xrange(1, 15001):

            load_timer.tic()
            images, labels = data.get()
            load_timer.toc()
            feed_dict = {net.images: images, net.labels: labels}
            # print step

            try:
                if(step%print_itr==0):

                    # print 'in print_itr'
                    train_timer.tic()
                    summary_str, loss, _,curr_lr = sess.run(
                        [summary_op, net.total_loss,train_op,learning_rate],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f} Speed: {:.3f}s/iter,'
                        ' Load: {:.3f}s/iter \n').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        data.epoch,
                        int(step),
                        curr_lr,
                        loss,
                        train_timer.average_time,
                        load_timer.average_time
                        )
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

            except Exception as e:

                log('EXCEPTION '+e,log_path)



                






def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    yolo = Original_Yolo_Model()
    pascal = pascal_voc('train','/home/vikram_mm/yolo_tensorflow/data/')

    

    print('Start training ...')
    train(yolo,pascal)
    print('Done training.')

if __name__ == '__main__':

    main()