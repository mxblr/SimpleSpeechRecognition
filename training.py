from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, Embedding, Dropout
from keras.models import Model, Sequential
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.callbacks import TensorBoard
from keras import metrics
from keras import optimizers

import numpy as np
import os
import json

import data_wrangler
import model_zoo
import data_read

######################### Model ######################### 
def my_model():
	"""
	Model to train

	Args:

	Returns:
	model: the Keras model
	"""
	model = Sequential()
	model.add(Dense(256, input_shape = (101, 128),activation='relu'))
	model.add(LSTM(64,return_sequences=True)) 
	model.add(LSTM(64, return_sequences=True)) 
	model.add(LSTM(32, return_sequences=True)) 
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(32, return_sequences=True)) 
	model.add(LSTM(32 ,return_sequences=False))
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	
	return model



######################### Data #########################

def load_dataset(dataset_size=-1, validation_set_size = -1):
	"""
	Loading the whole dataset into memory

	Args:
	dataset_size:			int used to load the desired number of files from the training set, defaults to the whole dataset
	validation_set_size:    int used to load the desired number of files from the validation set, defaults to the whole dataset

	Returns:
	x_train: training data
	x_test:  validation data
	y_train: training labels
	y_test:  validation labels
	"""
	

	x_train, x_test,y_train, y_test = [],[],[],[]


	#with open("training_set_v5.txt", "r") as f: 
	with open("training_set_mel.txt", "r") as f:
		x_train = json.load(f)
		if dataset_size > 0:
			x_train = x_train[:dataset_size]
		print(np.shape(x_train))
		y_train = [[[data_read.one_hot_encoding(data_read.find_label(filename))]]*1 for filename in x_train	] #TODO: 101
		x_train = [np.reshape(np.loadtxt(filename), (101, 128)) for filename in x_train]

		print(np.shape(y_train))
		x_train = np.array(x_train)
		y_train = np.reshape(np.array(y_train), (len(x_train),  12)) #TODO: (len(x_test),101, 12)
		print(np.shape(y_train))

	#with open("validation_set_v5.txt", "r") as f: 
	with open("validation_set_mel.txt", "r") as f:
		x_test = json.load(f)
		if validation_set_size > 0:
			x_test = x_test[:validation_set_size]
		y_test = [[[data_read.one_hot_encoding(data_read.find_label(filename))]]*1 for filename in x_test	] #TODO: 101
		x_test = [np.reshape(np.loadtxt(filename), (101, 128)) for filename in x_test]

		x_test = np.array(x_test)
		y_test = np.reshape(np.array(y_test), (len(x_test), 12)) #TODO: (len(x_test),101, 12)

	return ( x_test, y_test, x_train, y_train)


######################### Training ######################### 
def train_model(model, epochs,  x_test, y_test, x_train, y_train, weights_path, log_path, learning_rate):
	"""
	Training a model on training data, while evaluating it on the validation set

	Args:
	model: Keras model to train
	epochs: number of epochs the model should be trained for
	x_train: training data
	x_test:  validation data
	y_train: training labels
	y_test:  validation labels 
	weights_path: the filepath of weights of the model, if existant
	log_path: filepath of were to log data about training into
	learning_rate: learning rate used to train the model

	Returns:
	- 
	"""
	adam = optimizers.Adam( lr=learning_rate)
	sdg = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)

	model.compile( loss='categorical_crossentropy',
					optimizer=adam,
					metrics=['accuracy'])


	if os.path.isfile(weights_path):
		model.load_weights(weights_path, by_name=False)

	cb = callbacks.Callback()
	cb.set_model(model)
	tensorboard = TensorBoard(log_dir=log_path)
	tensorboard.set_model(model)
	checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
	#early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    

	callbacks_list = [checkpoint, cb, tensorboard]


	model.fit(x_train, y_train, batch_size=256, epochs=epochs, shuffle= True, verbose = 2,  callbacks=callbacks_list, validation_data=(x_test, y_test))

	model.save_weights(weights_path)


	score = model.evaluate(x_test, y_test, batch_size=128)
	print(score)

def predict(model, filename,weights_path):
	"""
	Predicts the label for a single file with MFCC features.

	Args:
	model: Keras model 
	weights_path: filepath to the weights of the model
	filename: filepath to the file which should be processed

	Returns:
	The label.
	"""
	if os.path.isfile(weights_path):
		model.load_weights(weights_path, by_name=False)
	
	sample = data_wrangler.single_data_processing(filename)
	pred = model.predict(sample)
	print(pred)
	print(data_read.label_from_oh(pred))
	return data_read.label_from_oh(pred)

def predict_with_mel_spectrogram(model, filename, weights_path):
	"""
	Predicts the label for a single file  with mel spectogram features.

	Args:
	model: Keras model 
	weights_path: filepath to the weights of the model
	filename: filepath to the file which should be processed

	Returns:
	The label.
	"""
	if os.path.isfile(weights_path):
		model.load_weights(weights_path, by_name=False)
	
	sample = data_wrangler.single_data_processing_with_mel_spectrogram(filename)
	pred = model.predict(sample)
	print(pred)
	print(data_read.label_from_oh(pred))

if __name__ == "__main__":
	
	model = my_model()
	x_test, y_test, x_train, y_train = load_dataset()

	d = {}
	for l in y_train:
		d[data_read.label_from_oh(l)] = d.get(data_read.label_from_oh(l), 0) +1
	print(d)

	
	log_path = "models/v54"
	weights_path = log_path+"/weights.best.hdf5"
	
	learning_rate = 0.0001
	train_model(model, 1000, x_test, y_test, x_train, y_train,weights_path,log_path, learning_rate)

	if os.path.isfile(weights_path):
		model.load_weights(weights_path, by_name=False)
	pred = model.predict(x_test,batch_size=128)
	x= 0
	for p_i, l_i in zip(pred, [data_read.label_from_oh(l) for l in y_test]):
		x += data_read.label_from_oh(p_i) == l_i
		print(np.amax(p_i), data_read.label_from_oh(p_i), l_i)
	print(x/len(pred))	
	

