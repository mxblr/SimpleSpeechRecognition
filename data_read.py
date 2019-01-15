__author__ = "Paul Schmidt-Barbo and Maximilian Beller"


###################################################################
#################### Import required packages #####################
###################################################################

import os
import numpy as np
import random
import json

###################################################################
#################### Import required packages #####################
###################################################################

def one_hot_encoding(label):
	"""
	Create a one hot encoding from a label.

	Args:
	label: 		the label of the current data file

	Returns:
	ohe:		one hot encoding of label
	"""
	label_list = ["yes", "no","up", "down", "left","right","on", "off","stop", "go","_background_noise_"]
	ohe 	= [label ==  label_list[0],  # yes
				label ==  label_list[1],  # no
				label ==  label_list[2],  # up
				label ==  label_list[3],  # down
				label ==  label_list[4],  # left
				label ==  label_list[5],  # right
				label ==  label_list[6],  # on
				label ==  label_list[7],  # off
				label ==  label_list[8],  # stop
				label ==  label_list[9],  # go
				label ==  label_list[10], # silence
				label not in label_list] # unknown
	return np.array(ohe)	

def label_from_oh(one_hot_encoding):
	"""
	Get the label from the one-hot-encoding

	Args:
	one_hot_encoding: 		one hot encoding of label

	Returns:
	label: 		the label of the current data file
	"""
	label_list = ["yes", "no","up", "down", "left","right","on", "off","stop", "go","silence", "unknown"]
	return(label_list[np.argmax(one_hot_encoding)])		

def splitDataset(paths, validation_set_size=0.1, filename_validation="validation_set.txt", filename_training="training_set.txt"):
	"""
	Split the dataset into training and validation set.

	Args:
	paths: 					list of paths
	validation_set_size: 	% of dataset that is used as validation set
	filename_validation: 	filename of file to save paths of validation set
	filename_training: 		filename of file to save paths of training set

	Returns:
	-
	"""
	random.shuffle(paths)
	validation_set 	= paths[:int(validation_set_size*len(paths))]
	training_set 	= paths[int(validation_set_size*len(paths)):]

	with open(filename_validation, "w") as f:
		json.dump(validation_set,f)
	with open(filename_training, "w") as f:
		json.dump(training_set,f)

def find_label(path):
	"""
	Gets the label from filepath

	Args:
	path: 		file path

	Returns:
	-
	"""
	path_components = path.split("/")
	label 			= path_components[-2]
	return label

if __name__ == "__main__":
	mfcc_Path = 'data/mfcc_v5/train/'
	melspec_Path = 'data/melspec/train/'
	paths = []
	with open("balanced_dataset_mel.txt", "r") as f:
		paths = json.load(f)
		
	print(len(paths))
	splitDataset(paths=paths, validation_set_size=0.15, filename_validation="validation_set_mel.txt", filename_training="training_set_mel.txt")		