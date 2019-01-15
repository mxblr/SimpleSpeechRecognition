import tensorflow as tf
import tflearn
import data_wrangler
import data_read
import numpy as np
import json
import random


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

path_tfrecords_train = "data/training_records_balanced/train.tfrecords"
path_tfrecords_valid = "data/validation_records_balanced/validation.tfrecords" 
label_list = ["yes", "no","up", "down", "left","right","on", "off","stop", "go","silence"]

def inspect_data(files):
	import data_batch
	"""
	Counting the number of files per class

	Args:
	files: TFRecord containing all files

	Returns:
	count_dict:  dictionary containing all classes as keys and the number of examples as values
	"""
	dataset_test = tf.data.TFRecordDataset(filenames=files)
	dataset_test = dataset_test.map(data_batch.parse)
	dataset_test = dataset_test.batch(1)

	# Create an iterator for the dataset and the above modifications.
	iterator_test = dataset_test.make_one_shot_iterator()

	#Get the next batch of images and labels.
	_, labels_raw = iterator_test.get_next()

	true_classes = []
	sess = tf.Session()
	while(True):
		try:
			labels = sess.run(labels_raw)
			labels = labels.astype('U13')
			label = labels[0]

			if("background" in label):
				true_classes.append("silence")
			elif(label not in label_list):				
				true_classes.append("unknown")
			else:
				true_classes.append(label)
		except Exception as e: 
			
			break


	true_classes = np.asarray(true_classes).flatten()
	print(true_classes.shape)
	count_dict = {}
	for i in label_list:
		count_dict[i] = 0
	count_dict["unknown"] = 0 

	for i in true_classes:
		count_dict[i] += 1

	return count_dict

def balance_files(paths):
	"""
	Equalizing the number of files per class.

	Args:
	paths: all file paths 

	Returns:
	class_paths:  array containing all files after balancing
	"""
	print("Inspect Audio Files to ensure a balanced dataset.")
	count_dict = {}
	for i in label_list:
		count_dict[i] = 0

	unknown_paths = []
	class_paths = []
	for file in paths:
		class_name = file.split("/")[-2]
		if("background" in class_name):
			count_dict["silence"] += 1
			class_paths.append(file)
		elif(class_name not in label_list):
			unknown_paths.append(file)
		else:
			count_dict[class_name] += 1
			class_paths.append(file)

	number_samples = [int(x) for x in count_dict.values()]
	mean_number_sample = int(np.mean(np.asarray(number_samples)))
	
	print("Mean number of samples from 'known' classes: %d.\n Shuffle files from other 'unknown' classes and select %d samples." % (mean_number_sample,mean_number_sample))
	random.shuffle(unknown_paths)
	class_paths += unknown_paths[:mean_number_sample]

	count_dict["unknown"] = mean_number_sample
	print(count_dict)
	return class_paths

if __name__ == "__main__":
	count_dict_train = inspect_data(path_tfrecords_train)
	count_dict_valid = inspect_data(path_tfrecords_valid)
	print(count_dict_train)
	print(count_dict_valid)
	counts_train = np.zeros(len(count_dict_train.keys()))
	counts_valid = np.zeros(len(count_dict_train.keys()))
	classes = []

	for idx,i in enumerate(count_dict_train.keys()):
		counts_train[idx] = count_dict_train[i]
		counts_valid[idx] = count_dict_valid[i]
		classes.append(i)
	

	def autolabel(rects):
		# attach some text labels´
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., height + 1,'%d' % float(height),ha='center', va='bottom', rotation=45)



	# Histogramm Training Data
	fig,ax = plt.subplots()
	y_pos = np.arange(len(classes))

	plt.title('Training-Data True Classes (%d Samples)' % np.sum(counts_train))
	plt.ylabel('Counts')
	plt.xlabel("Classes")


	c_rects=plt.bar(y_pos, counts_train, align='center', alpha=0.6)
	autolabel(c_rects)
	ax.set_xticks(range(len(classes)))
	ax.set_xticklabels(classes, rotation='vertical')
	plt.show()


	# Histogramm Validation Data
	fig,ax = plt.subplots()
	y_pos = np.arange(len(classes))

	plt.title('Validation-Data True Classes (%d Samples)' % np.sum(counts_valid))
	plt.ylabel('Counts')
	plt.xlabel("Classes")


	c_rects=plt.bar(y_pos, counts_valid, align='center', alpha=0.6)
	autolabel(c_rects)
	ax.set_xticks(range(len(classes)))
	ax.set_xticklabels(classes, rotation='vertical')
	plt.show()
