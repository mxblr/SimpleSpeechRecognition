from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, Embedding, Dropout
from keras.models import Model, Sequential
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.callbacks import TensorBoard
from keras import metrics
from keras import optimizers

####################################################
############### Unbalanced DATASET: ################
####################################################

# Model number in paper: not included into paper
def model_v34():
	model = Sequential()
	model.add(Dense(52, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(128,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(64, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(24 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	
	#686 epochs - learning_rate = 0.001
	#acc 94.73 / 28.97

	return model

# Model number in paper: not included into paper
def model_v35():
	model = Sequential()
	model.add(Dense(52, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(128,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(64, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32 ,return_sequences=False)) #TODO: TRUE
	model.add(LSTM(32 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(24 ,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(12, activation="softmax"))

	#930 epochs - learning_rate = 0.001
	#acc 61.03/ 22.49

	return model

####################################################
######### Balanced, unnormalized DATASET ###########
####################################################
# Model number in paper: 1
def model_v38():
	model = Sequential()
	model.add(Dense(52, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(64,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#535 epochs - learning_rate = 0.001
	#acc 98.49 / 65.71	

	return model

# Model number in paper: 2
def model_v39():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#98 epochs - learning_rate = 0.001
	#acc 81.45 / 65.07


	return model

# Model number in paper: 3
def model_v40():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#624 epochs - learning_rate = 0.005
	#acc 88.01 / 79.59

	#+63 epochs - learning_rate = 0.001
	#acc 99.31 / 78.18         

	#test acc: 61.677


	return model

# Model number in paper: 4
def model_v41():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16 ,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#128 epochs - learning_rate = 0.005
	#acc 82.32 / 71.47

	return model

# Model number in paper: 5
def model_v42():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#243 epochs - learning_rate = 0.005
	#acc 88.55 / 72.35

	return model

####################################################
#############	NORMALIZED DATASET #################
####################################################

# Model number in paper: 6
def model_v43():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#194 epochs - learning_rate = 0.001
	#acc 90.15 / 63.35

	return model

# Model number in paper: 7
def model_v44():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(16 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#158 epochs - learning_rate = 0.002
	#acc 87.37 / 65.25

	return model

# Model number in paper: 8
def model_v45():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	
	#231 epocs - learning_rate = 0.002
	#acc 91.05 / 65.88

	return model


# Model number in paper: 9
def model_v46():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 26),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#1000 epochs - learning_rate = 0.004
	#acc = 90.00 / 74.00
	#test-test acc = 0.63432

	return model

####################################################
################ ENRICHED DATASET ##################
####################################################

# Model number in paper: 10
def model_v47():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 13),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#1000 epochs - learning_rate = 0.004
	#acc = 88.00 / 79.00
	#test-test acc = 0.____


	return model


# Model number in paper: 11
def model_v48():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 13),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#207 epochs - learning_rate=0.005
	#acc 83.26 / 79.03

	return model

# Model number in paper: 12
def model_v49():
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

	#168 epochs - learning_rate= 0.005
	#acc 85.34 / 80.28
	#+16 epochs -learning_rate = 0.001
	#acc 91.00/81.80

	return model


# Model number in paper: 13
def model_v50():
	#= v49_1 + 293 epochs - sdg learning_rate 0.001
	#acc 89.46 / 82.14
	model = model_v49()

	return model

# Model number in paper: 14
def model_v51():
	#= v49_1 + 256 epochs - sdg learning_rate 0.004
	#acc 90.45 / 82.01
	model = model_v49()

	return model


####################################################
################# LOG SPECTRUM DATASET #############
####################################################

# Model number in paper: not included into paper
def model_v52():
	model = Sequential()
	model.add(Dense(39, input_shape = (101, 128),activation='relu'))
	model.add(LSTM(48,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(36, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(24 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#81 epochs -learning_rate = 0.004
	#acc 8.00/8.00

	return model

# Model number in paper: not included into paper
def model_v53():
	model = Sequential()
	model.add(Dense(256, input_shape = (101, 128),activation='relu'))
	model.add(LSTM(64,return_sequences=True)) #TODO: TRUE
	model.add(LSTM(64, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32, return_sequences=True)) #TODO: TRUE
	model.add(LSTM(32 ,return_sequences=False)) #TODO: TRUE
	model.add(Dense(18 ,activation='relu'))
	model.add(Dense(12, activation="softmax"))

	#250 epochs - learning_rate = 0.005
	#acc 8.00 / 8.00

	return model


def model_v54():
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

	#___ epochs - learning_rate = 0.001
	# acc

	return model