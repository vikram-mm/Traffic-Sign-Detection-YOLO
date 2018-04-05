import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from btsd_reader import btsd_reader
import cv2
import os

reader = btsd_reader('train', '/home/vikram_mm/minor_project/data')

learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 16, 16, 3), name='inputs')
# targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv1')
# Now 64x64x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool1')
# Now 32x32x32
# conv2a = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2a')
# conv2b = tf.layers.conv2d(inputs=conv2a, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2b')
# # Now 32x32x32
# maxpool2 = tf.layers.max_pooling2d(conv2b, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool2')
# # Now 16x16x32
# conv3a = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv3a')
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv2')
# Now 16x16x16
encoded = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same', name='ae_maxpool2')
# Now 8x8x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 16x16x16
# conv4a = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4a')
# conv4b = tf.layers.conv2d(inputs=conv4a, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4b')
# # Now 16x16x16
# upsample2 = tf.image.resize_images(conv4b, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 32x32x16
conv3 = tf.layers.conv2d(inputs=upsample1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv3')
# conv5b = tf.layers.conv2d(inputs=conv5a, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv5b')
# Now 32x32x32
upsample2 = tf.image.resize_images(conv3, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, )
# Now 64x64x32
conv4 = tf.layers.conv2d(inputs=upsample2, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, name='ae_conv4')
# Now 64x64x32
logits = tf.layers.conv2d(inputs=conv4, filters=3, kernel_size=(3,3), padding='same', activation=None, name='ae_conv5')
#Now 64x64x3
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
# Pass logits through sigmoid and calculate the cross-entropy loss
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# # Get cost and define the optimizer
# cost = tf.reduce_mean(loss)
# opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


loss_op = tf.reduce_mean(tf.square(decoded - inputs_))
tf.summary.scalar('loss',loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
sess = tf.Session()
merged = tf.summary.merge_all()
saver = tf.train.Saver()
epochs = 200
batch_size = 45
# Set's how much noise we're adding to the MNIST images
# noise_factor = 0.5
sess.run(tf.global_variables_initializer())
# for e in range(epochs):
ctr=0
iter=0
train_writer = tf.summary.FileWriter("saves/AE/model5/",
                                    sess.graph)
while(reader.epoch<=epochs):
	# for ii in range(mnist.train.num_examples//batch_size):
		# batch = mnist.train.next_batch(batch_size)
		# # Get images from the batch
		# imgs = batch[0].reshape((-1, 28, 28, 1))
		
		# # Add random noise to the input images
		# noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
		# # Clip the images to be between 0 and 1
		# noisy_imgs = np.clip(noisy_imgs, 0., 1.)
		# tf.summary.scalar('losstf.summary.scalar('loss',loss_op)',loss_op)
		# Noisy images as inputs, original images as targets

	images = reader.get()
	batch_cost, _, summary = sess.run([loss_op, train_op,merged], feed_dict={inputs_: images})
	train_writer.add_summary(summary, iter)

	'''print("Epoch: {}/{}...".format(reader.epoch, epochs),
			  "Training loss: {:.4f}".format(batch_cost))'''

	# if ctr==10:
	# 	recon = sess.run(decoded, feed_dict={inputs_: images})
	# 	print type(images[10]), type(recon[10])
	# 	print recon[10].shape
	# 	# print images[10]
	# 	cv2.imwrite('input.png',images[10])
	# 	cv2.imwrite('output.png', recon[10])
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	if reader.epoch == 100:
		print "Decreasing lr"
		learning_rate=0.0001

	ctr+=1
	iter+=1
	if iter%500 == 0:

		print("Epoch: {}/{}...".format(reader.epoch, epochs),"iter : {}".format(iter),
			  "Training loss: {:.4f}".format(batch_cost))
		save_path = saver.save(sess, os.path.join('saves','AE','model5','save.ckpt'))
