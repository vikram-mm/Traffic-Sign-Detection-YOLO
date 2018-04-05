import tensorflow as tf
from btsd import btsd_reader
import cv2
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = ""

reader = btsd_reader('train', '/home/vikram_mm/minor_project/data')
learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 448, 448, 3), name='inputs')
# targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv1')
# Now 448x448x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool1')
# Now 224x224x32
# conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2')
conv2a = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2a')
conv2b = tf.layers.conv2d(inputs=conv2a, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2b')
# Now 224x224x32
maxpool2 = tf.layers.max_pooling2d(conv2b, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool2')
# Now 112x112x32
# conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv3')
conv3a = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv3a')
conv3b = tf.layers.conv2d(inputs=conv3a, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv3b')
# Now 112x112x32
encoded = tf.layers.max_pooling2d(conv3b, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool3')
# Now 56x56x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(112,112), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 112x112x16
# conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4')
conv4a = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4a')
conv4b = tf.layers.conv2d(inputs=conv4a, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4b')
# Now 112x112x16
upsample2 = tf.image.resize_images(conv4b, size=(224,224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 224x224x16
# conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv5')
conv5a = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv5a')
conv5b = tf.layers.conv2d(inputs=conv5a, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv5b')

# Now 224x224x32
upsample3 = tf.image.resize_images(conv5b, size=(448,448), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 448x448x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv6')
# Now 64x64x32
logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None, name='ae_conv7')
#Now 64x64x3
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
saver = tf.train.Saver()
images, _ = reader.get()
# print (images)
# exit(0)
with tf.Session() as sess:
  # Restore variables from disk.
	saver.restore(sess, "saves/AE/model2/save.ckpt")
	print("Model restored.")
  # if ctr==10:
	recon = sess.run(decoded, feed_dict={inputs_: images})
	print type(images[10]), type(recon[10])
	print recon[10].shape
	# print images[10]
	cv2.imshow('input.png',images[10])
	cv2.imshow('output.png', recon[10])
	cv2.imshow('input1.png',images[20])
	cv2.imshow('output1.png', recon[20])
	cv2.imshow('input2.png',images[30])
	cv2.imshow('output2.png', recon[30])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

  
  # Check the values of the variables
  
