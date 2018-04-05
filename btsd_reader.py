import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy

class btsd_reader(object):
	def __init__(self, phase,data_path,batch_size=45,rebuild=False,flipped=True):

		self.data_path = os.path.join(data_path, 'Training')
		self.cache_path = os.path.join(data_path, 'ae_cache')
		self.batch_size = batch_size
		self.image_size = 16
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
		count = 0
		while count < self.batch_size:
			imname = self.image_list[self.cursor][0]
			# flipped = self.gt_labels[self.cursor]['flipped']
			# print imname, self.image_list[self.cursor][1]
			images[count, :, :, :] = self.image_read(imname, flipped=self.image_list[self.cursor][1])
			count += 1
			self.cursor += 1
			if self.cursor >= len(self.image_list):
				np.random.shuffle(self.image_list)
				self.cursor = 0
				self.epoch += 1
		return images


	def image_read(self, imname, flipped=False):
		image = cv2.imread(imname)
		image = cv2.resize(image, (self.image_size, self.image_size))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		image = (image / 255.0) #* 2.0 - 1.0
		if flipped:
			image = image[:, ::-1, :]
		return image



	def prepare(self):
		image_list = []
		for root, directories, filenames in os.walk(self.data_path):
			for filename in filenames: 
				if filename.endswith(".ppm"):
					temp_list = []
					image_list.append([os.path.join(root,filename), True])
					image_list.append([os.path.join(root,filename), False])
					
		np.random.shuffle(image_list)
		print "Dataset loaded and shuffled"
		print "No of images in dataset is ",len(image_list)
		self.image_list = image_list
		return image_list

if __name__ == '__main__':
	reader = btsd_reader('train', 'data')
	a = reader.get()
	# print a[0]
	# cv2.imshow('checking.png',a[0])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# print "done"
	# reader.get()