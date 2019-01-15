from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, Embedding, Dropout
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
import json
from keras import metrics
from keras import optimizers
import data_read
from keras import layers
import os
import h5py
import data_wrangler
from keras.callbacks import ModelCheckpoint
from keras import callbacks
import random
import csv
import math
import sys
import model_zoo

def my_model():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 13),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))
	
	return model


def predict_unpreprocessed(weights_path, file_paths, output_filename):
	"""
	Predict list of raw '.wav' files and saves it to CSV file.

	Args:
	weights_path:		weights for model
	file_paths:			files to predict a label for
	output_filename: 	name of CSV file

	Returns:
	-
	"""
	if os.path.isfile(weights_path):
		print("Loading weights")
		model.load_weights(weights_path, by_name=False)

	with open(output_filename, "w") as csv_file:
	
		csv_writer = csv.writer(csv_file, delimiter=",", dialect="unix", quoting = csv.QUOTE_NONE)

		iteration = 0
		max_iterations = math.ceil(len(file_paths)/256)

		while len(file_paths) >0:
			batch_filenames = file_paths[:256]

			batch = [data_wrangler.single_data_processing(filename) for filename in batch_filenames]
			batch = np.reshape(np.array(batch), (len(batch),101,13))

			pred = model.predict(batch, batch_size=256)
			for filename, prediction in zip(batch_filenames, pred):
				clean_filename = filename.split("/")[-1]
				csv_writer.writerow([clean_filename, data_read.label_from_oh(prediction)]) 																	# TODO
			
			file_paths = file_paths[min(len(batch),256):]
			iteration = iteration+1
			sys.stdout.write('\r>> Predicting classes for batch %d of %d.' % (iteration,  max_iterations))
			sys.stdout.flush()

def predict_single_unpreprocessed(weights_path, file, model):
	"""
	Predict list of raw '.wav' files and saves it to CSV file.

	Args:
	weights_path:		weights for model
	file_paths:			files to predict a label for
	output_filename: 	name of CSV file

	Returns:
	-
	"""
	if os.path.isfile(weights_path):
		#print("Loading weights")
		model.load_weights(weights_path, by_name=False)

	batch = [data_wrangler.single_data_processing_raw(file)]
	batch = np.reshape(np.array(batch), (len(batch),101,13))


	pred = model.predict(batch, batch_size=256)
	print(data_read.label_from_oh(pred))

def predict_preprocessed(weights_path, file_paths, output_filename):
	"""
	Predict list processed files and saves it to CSV file.

	Args:
	weights_path:		weights for model
	file_paths:			files to predict a label for
	output_filename: 	name of CSV file

	Returns:
	-
	"""
	if os.path.isfile(weights_path):
		print("Loading weights")
		model.load_weights(weights_path, by_name=False)

	with open(output_filename, "w") as csv_file:
	
		csv_writer = csv.writer(csv_file, delimiter=",", dialect="unix", quoting = csv.QUOTE_NONE)

		iteration = 0
		max_iterations = math.ceil(len(file_paths)/256)

		csv_writer.writerow(["fname","label"])

		while len(file_paths) > 0:

			batch = file_paths[:256]
			batch = [np.reshape(np.loadtxt(filename), (101, 128)) for filename in batch]
			batch = np.asarray(batch)

			pred = model.predict(batch, batch_size=256)
			for filename, prediction in zip(file_paths, pred):
				clean_filename = filename.split("/")[-1].split(".")[0]+".wav"
				csv_writer.writerow([clean_filename, data_read.label_from_oh(prediction)]) 	
			
			file_paths=file_paths[len(batch):]																# TODO
			iteration +=1
			print("%s/%s done." % (iteration, max_iterations))	
			

if __name__ == "__main__":
	model = model_zoo.model_v54()

	file_list = []

	for f in os.listdir("data/melspec/test/audio"):
		file_list.append("data/melspec/test/audio/"+f)


	version = "v54"
	weights_path = "models/%s/weights.best.hdf5" % (version)
	prediction_filename = "predictions/prediction_%s.csv" % (version)
	predict_preprocessed(weights_path,file_list, prediction_filename )