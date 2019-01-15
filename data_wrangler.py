__author__ = "Paul Schmidt-Barbo and Maximilian Beller"

###################################################################
#################### Import required packages #####################
###################################################################

import os
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
import concurrent.futures
import math
import data_balancing
import json

###################################################################
######################## Methods ##################################
###################################################################


def cutAudio(samples, length=16000):
	"""
	Method to cut audio into chunks of length 'length'.

	Args:
	samples: 	audio samples
	length: 	desired length

	Returns:
	cuts:		cut up samples
	"""
	cuts = []
	while samples != []:
		if len(samples) <length:
			cuts.append(samples)
			samples = []
		else:
			cuts.append(samples[:length])	
			samples = samples[length:]
	return cuts



def cutAudioRandom(samples,number_cuts,length=16000):
	"""
	Method to randomly cut audio into chunks of length 'length'. 

	Args:
	samples: 	audio samples
	length: 	desired length
	number_cuts: number of cuts from the original sample

	Returns:
	cuts:		cut up samples
	"""
	cuts = []
	for i in range(number_cuts):
		random_start = np.random.random_integers(low = 0, high = len(samples)-(length))
		cuts.append(samples[random_start:(random_start+length)])
	return cuts

def loadAudio(filepath):
	"""
	Method to load an audio file as an array. 

	Args:
	filepath: 	filepath to ".wav" file

	Returns:
	sample_rate:	rate, with which audio was recored. Usually 16000Hz
	samples:		Sample of audio file
	"""
	sample_rate, samples = wavfile.read(filepath)
	return sample_rate, samples

def resample(sample_rate, samples, new_sample_rate=8000):
	"""
	Method to resample an audio file 

	Args:
	sample_rate:		original rate, with which audio was recored. Usually 16000Hz
	samples:			Sample of audio file
	new_sample_rate: 	new desired sample-rate

	Returns:
	resampled: 			audio file with new sample rate
	"""
	resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
	return resampled

def padAudio(samples, length):
	"""
	Method to pad a audio file. Can be used to make all files equal length.

	Args:
	samples:			Sample of audio file
	length: 			new desired length of the file

	Returns:
	samples: 			audio file with length 'length'
	"""
	dif = length - len(samples)
	if dif > 0:
		a = [0]*dif
		samples = np.append(samples,a)
	return samples	

def melspectrogram(samples, sample_rate, number_features=128): 
	"""
	Method calculating the log power of the melspectrogram of an audio file.
	Overlapping windows of length 20ms and an overlap of 10 ms are used.

	Args:
	samples:			Sample of audio file
	sample_rate:		original rate, with which audio was recored. Usually 16000Hz
	number_features: 	number of melspectrogram featueres

	Returns:
	log_S: 				featueres of spectorgram.
	"""	
	S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=number_features,hop_length=int(0.01*sample_rate), n_fft = int(0.02*sample_rate))
	log_S = librosa.power_to_db(S, ref=np.max) # log power required? ---> motivated by human hearing: we don't hear loudness on a linear scale.
	return log_S

def mfcc(samples, sample_rate, n_mfcc=13, n_mels=24):
	"""
	Method calculating the MFCC from the log power of the melspectrogram of an audio file.
	Overlapping windows of length 20ms and an overlap of 10 ms are used.

	Args:
	samples:			Sample of audio file
	sample_rate:		original rate, with which audio was recored. Usually 16000Hz
	n_mfcc: 			number of MFCC featueres
	n_mels:				number of melspectrogram featueres

	Returns:
	mfcc: 				featueres of MFCC.
	"""		
	log_S = melspectrogram(samples, sample_rate, number_features=n_mels)	
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
	
	# Let's pad on the first and second deltas while we're at it
	#delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	return mfcc

      

def normalization(mfcc_paths, n_mfcc=13):
	"""
	Method to normalize features by calculating mean and standard deviation of whole data set.

	Args:
	mfcc_paths:			Sample of audio file
	n_mfcc:		original rate, with which audio was recored. Usually 16000Hz
	

	Returns:
	mean 				mean of dataset
	std: 				standard deviation of dataset
	"""		

	shape = (n_mfcc,101)
	mfcc_all = np.zeros(shape)
	number = 0
	skipped = 0
	skipped_paths = []
	for mfcc_i in mfcc_paths:
		sample 		= np.loadtxt(mfcc_i)

		if sample.shape == shape:
			mfcc_all 	= np.add(mfcc_all, sample)
			number += 1
		else:
			skipped += 1

			skipped_paths += [mfcc_i]

	print("%s frames skipped, because they were to short." % skipped)		
	print(skipped_paths)
	
	mean = np.divide(np.array(mfcc_all), number)
	std_all = np.zeros(shape)
	for mfcc_i in mfcc_paths:
		sample 		= np.loadtxt(mfcc_i)
		if sample.shape == shape:
			std_all 	= np.add(std_all,np.subtract(sample, mean))

	std = np.divide(np.array(std_all), number)

	np.savetxt("mean.txt", mean)
	np.savetxt("std.txt", std)

	mfcc_path_1 = mfcc_paths[:math.floor(len(mfcc_paths)/4)]
	mfcc_path_2 = mfcc_paths[math.floor(len(mfcc_paths)/4):math.floor(len(mfcc_paths)/2)]
	mfcc_path_3 = mfcc_paths[math.floor(len(mfcc_paths)/2):math.floor(len(mfcc_paths)/2)+math.floor(len(mfcc_paths)/4)]
	mfcc_path_4 = mfcc_paths[math.floor(len(mfcc_paths)/2)+math.floor(len(mfcc_paths)/4):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(single_normalization, mfcc_path_1, mean, std)
		path_2 = executor.submit(single_normalization, mfcc_path_2, mean, std)
		path_3 = executor.submit(single_normalization, mfcc_path_3, mean, std)
		path_4 = executor.submit(single_normalization, mfcc_path_4, mean, std)	

	return [mean, std]

def single_normalization(mfcc_paths, mean, std):
	"""
	Sub-Method to normalize features by shifting each data point by  mean and standard deviation of whole data set.

	Args:
	mean 				mean of dataset
	std: 				standard deviation of dataset
	

	Returns:
	-					just saves the files 
	"""		
	for mfcc_i in mfcc_paths:
		sample 	= np.loadtxt(mfcc_i)
		if sample.shape == shape:
			sample 	= np.divide(np.subtract(sample, mean), std)
			np.savetxt(mfcc_i, sample)

def single_data_processing(filename, sample_rate_global=16000):
	"""
	Method to process a single file. Used for prediction of new files.

	Args:
	filename 				path to ".wav" file 
	sample_rate_global: 	desired sample rate
	

	Returns:
	mfccA:					MFCC features of file
	"""	
	sample_rate, samples = loadAudio(filename)
	samples = resample(16000,samples, sample_rate_global)

	samples = padAudio(samples, sample_rate_global)	
	mfccA 	= mfcc(samples, sample_rate_global, n_mfcc=26)

	mean = np.loadtxt("mean.txt")
	std = np.loadtxt("std.txt")

	if (len(mfccA[0]) <= 101) and False:
		sample 	= np.divide(np.subtract(mfccA, mean), std)
		sample_shape = np.shape(sample)
		sample = np.reshape(sample, (sample_shape[1], sample_shape[0]))	
		return sample		
	return mfccA

def single_data_processing_raw(samples, sample_rate_global=16000):
	"""
	Method to process a single file. Used for prediction of new files.

	Args:
	filename 				path to ".wav" file 
	sample_rate_global: 	desired sample rate
	

	Returns:
	mfccA:					MFCC features of file
	"""	
	
	samples = resample(16000,samples, sample_rate_global)

	samples = padAudio(samples, sample_rate_global)	
	mfccA 	= mfcc(samples, sample_rate_global, n_mfcc=13)

	mean = np.loadtxt("mean.txt")
	std = np.loadtxt("std.txt")
	

	if (len(mfccA[0]) <= 101) and False:
		sample 	= np.divide(np.subtract(mfccA, mean), std)
		sample_shape = np.shape(sample)
		sample = np.reshape(sample, (sample_shape[1], sample_shape[0]))	
		return sample		
	return mfccA

def preprocess(wav_paths, mfcc_Path, n_mfcc, sample_rate_global):
	"""
	Method to process a dataset. Used for prediction of new files.

	Args:
	wav_paths 				Paths to original ".wav" files
	mfcc_Path: 				Directory to save mfcc featueres
	n_mfcc:					number of mfcc features
	sample_rate_global:		desired sample rate of data

	Returns:
	paths:					New paths to MFCC features
	"""	
	print("Converting audio files to MFCC files. \nSampling rate gets reduced to 8000Hz. \nLong chunks get cut down to length: 1 second.")
	shape = (n_mfcc,101)
	skipped_paths = []
	paths = []
	for audio_file in wav_paths:
		if(audio_file.split(".")[1] == "wav"):
			directory = audio_file.split("/")[-2] 
			filename = audio_file.split("/")[-1].split(".")[0]
			sample_rate, samples = loadAudio(audio_file)
				#samples =resample(16000,samples, sample_rate_global
			if len(samples) > sample_rate_global:
				sample_array = cutAudio(samples, sample_rate_global)
				enhanced_number_silence_samples = 5*len(sample_array)
				random_sample_array = cutAudioRandom(samples,enhanced_number_silence_samples,sample_rate_global) # after analysing the splits without random cutting increase noise data by 5 random cuts for each file 
				
				enhanced_sample_array = sample_array + random_sample_array
				
				for i,s in enumerate(enhanced_sample_array):
					s = padAudio(s, sample_rate_global)
					mfccA = mfcc(s, sample_rate_global, n_mfcc=n_mfcc)
					filename_i = str(mfcc_Path+"/"+directory+"/"+filename+str(i)+".txt")
					
					if mfccA.shape == shape:
						np.savetxt(filename_i, mfccA)
						paths += [filename_i]
					else:
						skipped_paths += [filename_i]
			else:
			
				samples = padAudio(samples, sample_rate_global)	
				
				mfccA = mfcc(samples, sample_rate_global, n_mfcc=n_mfcc)
				
				filename = str(mfcc_Path+"/"+directory+"/"+filename+".txt")	
				
				if mfccA.shape == shape:
					np.savetxt(filename, mfccA)
					paths += [filename]
				else:
					skipped_paths += [filename]
				
				
	print("Skipped files with wrong shape.")		
	print(skipped_paths)
	print("Process finished")
	return paths


def preprocess_with_mel_spectrogram(wav_paths, melspec_Path, n_mels, sample_rate_global):
	"""
	Method to process a dataset. Used for prediction of new files.

	Args:
	wav_paths 				Paths to original ".wav" files
	melspec_Path: 				Directory to save mfcc featueres
	n_mels:					number of mfcc features
	sample_rate_global:		desired sample rate of data

	Returns:
	paths:					New paths to Melspectrogram features
	"""	
	print("Converting audio files to Melspectrogram files. \nSampling rate gets reduced to 8000Hz. \nLong chunks get cut down to length: 1 second.")
	shape = (n_mels,101)
	skipped_paths = []
	paths = []
	for audio_file in wav_paths:
		if(audio_file.split(".")[1] == "wav"):
			directory = audio_file.split("/")[-2] 
			filename = audio_file.split("/")[-1].split(".")[0]
			sample_rate, samples = loadAudio(audio_file)
				#samples =resample(16000,samples, sample_rate_global
			if len(samples) > sample_rate_global:
				sample_array = cutAudio(samples, sample_rate_global)
				enhanced_number_silence_samples = 5*len(sample_array)
				random_sample_array = cutAudioRandom(samples,enhanced_number_silence_samples,sample_rate_global) # after analysing the splits without random cutting increase noise data by 5 random cuts for each file 
				
				enhanced_sample_array = sample_array + random_sample_array
				
				for i,s in enumerate(enhanced_sample_array):
					s = padAudio(s, sample_rate_global)
					melA = melspectrogram(s, sample_rate_global, number_features=n_mels)
					filename_i = str(melspec_Path+"/"+directory+"/"+filename+str(i)+".txt")
					
					if melA.shape == shape:
						np.savetxt(filename_i, melA)
						paths += [filename_i]
					else:
						skipped_paths += [filename_i]
			else:
			
				samples = padAudio(samples, sample_rate_global)	
				
				melA = melspectrogram(samples, sample_rate_global, number_features=n_mels)
				
				filename = str(melspec_Path+"/"+directory+"/"+filename+".txt")	
				
				if melA.shape == shape:
					np.savetxt(filename, melA)
					paths += [filename]
				else:
					skipped_paths += [filename]
				
				
	print("Skipped files with wrong shape.")		
	print(skipped_paths)
	print("Process finished")
	return paths

def normalization_with_mel_spectrorgram(melspec_paths, n_mels=128):
	"""
	Method to normalize features by calculating mean and standard deviation of whole data set.

	Args:
	melspec_paths:			Sample of audio file
	n_mels:		original rate, with which audio was recored. Usually 16000Hz
	

	Returns:
	mean 				mean of dataset
	std: 				standard deviation of dataset
	"""		

	shape = (n_mels,101)
	melspec_all = np.zeros(shape)
	number = 0
	skipped = 0
	skipped_paths = []
	for melspec_i in melspec_paths:
		sample 		= np.loadtxt(melspec_i)

		if sample.shape == shape:
			melspec_all 	= np.add(melspec_all, sample)
			number += 1
		else:
			skipped += 1

			skipped_paths += [melspec_i]

	print("%s frames skipped, because they were to short." % skipped)		
	print(skipped_paths)
	
	mean = np.divide(np.array(melspec_all), number)
	std_all = np.zeros(shape)
	for melspec_i in melspec_paths:
		sample 		= np.loadtxt(melspec_i)
		if sample.shape == shape:
			std_all 	= np.add(std_all,np.subtract(sample, mean))

	std = np.divide(np.array(std_all), number)

	np.savetxt("mean_mel.txt", mean)
	np.savetxt("std_mel.txt", std)

	melspec_path_1 = melspec_paths[:math.floor(len(melspec_paths)/4)]
	melspec_path_2 = melspec_paths[math.floor(len(melspec_paths)/4):math.floor(len(melspec_paths)/2)]
	melspec_path_3 = melspec_paths[math.floor(len(melspec_paths)/2):math.floor(len(melspec_paths)/2)+math.floor(len(melspec_paths)/4)]
	melspec_path_4 = melspec_paths[math.floor(len(melspec_paths)/2)+math.floor(len(melspec_paths)/4):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_1, mean, std)
		path_2 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_2, mean, std)
		path_3 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_3, mean, std)
		path_4 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_4, mean, std)	

	return [mean, std]



def single_normalization_with_mel_spectrogram(melspec_paths, mean, std):
	"""
	Sub-Method to normalize features by shifting each data point by  mean and standard deviation of whole data set.

	Args:
	mean 				mean of dataset
	std: 				standard deviation of dataset
	

	Returns:
	-					just saves the files 
	"""		
	for melspec_i in melspec_paths:
		sample 	= np.loadtxt(melspec_i)
		if sample.shape == shape:
			sample 	= np.divide(np.subtract(sample, mean), std)
			np.savetxt(melspec_i, sample)

def single_data_processing_with_mel_spectrogram(filename, sample_rate_global=16000):
	"""
	Method to process a single file. Used for prediction of new files.

	Args:
	filename 				path to ".wav" file 
	sample_rate_global: 	desired sample rate
	
	
	Returns:
	melspecA:					Melspectrogram features of file
	"""	
	sample_rate, samples = loadAudio(filename)
	samples = resample(16000,samples, sample_rate_global)

	samples = padAudio(samples, sample_rate_global)	
	melspecA 	= melspectrogram(samples, sample_rate_global, number_features=128)

	mean = np.loadtxt("mean_mel.txt")
	std = np.loadtxt("std_mel.txt")

	if (len(melspecA[0]) <= 101) and False:
		sample 	= np.divide(np.subtract(melspecA, mean), std)
		sample_shape = np.shape(sample)
		sample = np.reshape(sample, (sample_shape[1], sample_shape[0]))	
		return sample		
	return melspecA
	mel

if __name__ == "__main__":
############################################################################################
##############################  Preprocessing of train  set ################################
############################################################################################

	audio_path = 'data/train/audio'
	mfcc_Path = 'data/mfcc_v5/train'
	n_mfcc = 13
	sample_rate_global = 16000
	
	"""
	# Create Direcotries to save mfcc_s
	# Find all subfolder (Labels) in the dataset
	print("Creating directories.")
	subFolderList = []
	for x in os.listdir(audio_path):
		if os.path.isdir(audio_path + '/' + x):
			subFolderList.append(x)

	if not os.path.exists(mfcc_Path):
		os.makedirs(mfcc_Path)

	subFolderList = []
	for x in os.listdir(audio_path):
		if os.path.isdir(audio_path + '/' + x):
			subFolderList.append(x)
			if not os.path.exists(mfcc_Path + '/' + x):
				os.makedirs(mfcc_Path +'/'+ x)

		
		
	wav_paths = []
	for directory in os.listdir(audio_path):
		for audio_file in os.listdir(audio_path+"/"+directory):
			wav_paths += [audio_path+"/"+directory+"/"+audio_file]
		
	wav_path_1 = wav_paths[:math.floor(len(wav_paths)/4)]
	wav_path_2 = wav_paths[math.floor(len(wav_paths)/4):math.floor(len(wav_paths)/2)]
	wav_path_3 = wav_paths[math.floor(len(wav_paths)/2):math.floor(len(wav_paths)/2)+math.floor(len(wav_paths)/4)]
	wav_path_4 = wav_paths[math.floor(len(wav_paths)/2)+math.floor(len(wav_paths)/4):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(preprocess, wav_path_1, mfcc_Path, n_mfcc, sample_rate_global)
		path_2 = executor.submit(preprocess, wav_path_2, mfcc_Path, n_mfcc, sample_rate_global)
		path_3 = executor.submit(preprocess, wav_path_3, mfcc_Path, n_mfcc, sample_rate_global)
		path_4 = executor.submit(preprocess, wav_path_4, mfcc_Path, n_mfcc, sample_rate_global)				
			

		
	paths= []
	for directory in os.listdir(mfcc_Path):
		for mfcc_file in os.listdir(mfcc_Path+"/"+directory):
			paths += [mfcc_Path+"/"+directory+"/"+mfcc_file]


	print("Start normalization of data.")
	balanced_paths = data_balancing.balance_files(paths)
	with open("balanced_dataset.txt", "w") as f:
		json.dump(balanced_paths,f)
	mean, std = normalization(balanced_paths, n_mfcc=n_mfcc)
	 	print("Processing finished.")


	# ############################################################################################
	# #################################  Preprocessing of test set ###############################
	# ############################################################################################
		

	mean = np.loadtxt("mean.txt")
	std = np.loadtxt("std.txt")
		
	test_path = "data/test/audio/"
	mfcc_Path = "data/mfcc_v5/test/"
	test_files = []

	if not os.path.exists(mfcc_Path):
		os.makedirs(mfcc_Path)
		
	for f in os.listdir(test_path):
		test_files.append(test_path+f)
		
	wav_path_1 = test_files[:math.floor(len(test_files)/4)]
	wav_path_2 = test_files[math.floor(len(test_files)/4):math.floor(len(test_files)/2)]
	wav_path_3 = test_files[math.floor(len(test_files)/2):math.floor(len(test_files)/2)+math.floor(len(test_files)/4)]
	wav_path_4 = test_files[math.floor(len(test_files)/2)+math.floor(len(test_files)/4):]

	print("files distributed")

		#preprocess(test_files, mfcc_Path, n_mfcc, sample_rate_global)

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(preprocess, wav_path_1, mfcc_Path, n_mfcc, sample_rate_global)
		path_2 = executor.submit(preprocess, wav_path_2, mfcc_Path, n_mfcc, sample_rate_global)
		path_3 = executor.submit(preprocess, wav_path_3, mfcc_Path, n_mfcc, sample_rate_global)
		path_4 = executor.submit(preprocess, wav_path_4, mfcc_Path, n_mfcc, sample_rate_global)				
		
	print("files processed")
	mfcc_paths =[]
	for f in os.listdir(mfcc_Path):
		mfcc_paths.append(mfcc_Path+f)

	print(len(mfcc_paths))

	print("start normalization")
	mfcc_path_1 = mfcc_paths[:math.floor(len(mfcc_paths)/4)]
	mfcc_path_2 = mfcc_paths[math.floor(len(mfcc_paths)/4):math.floor(len(mfcc_paths)/2)]
	mfcc_path_3 = mfcc_paths[math.floor(len(mfcc_paths)/2):math.floor(len(mfcc_paths)/2)+math.floor(len(mfcc_paths)/4)]
	mfcc_path_4 = mfcc_paths[math.floor(len(mfcc_paths)/2)+math.floor(len(mfcc_paths)/4):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(single_normalization, mfcc_path_1, mean, std)
		path_2 = executor.submit(single_normalization, mfcc_path_2, mean, std)
		path_3 = executor.submit(single_normalization, mfcc_path_3, mean, std)
		path_4 = executor.submit(single_normalization, mfcc_path_4, mean, std)	

	print("finished")



############################################################################################
##############  Preprocessing of train  set with melspectrogram ############################
############################################################################################

	audio_path = 'data/train/audio'
	melspec_Path = 'data/melspec/train'
	n_mels = 128
	sample_rate_global = 16000

	#Create Direcotries to save melspectrograms
	#Find all subfolder (Labels) in the dataset
	print("Creating directories.")
	subFolderList = []
	for x in os.listdir(audio_path):
		if os.path.isdir(audio_path + '/' + x):
			subFolderList.append(x)

	# Create directory to save spectograms to
	if not os.path.exists(melspec_Path):
		os.makedirs(melspec_Path)

	subFolderList = []
	for x in os.listdir(audio_path):
		if os.path.isdir(audio_path + '/' + x):
			subFolderList.append(x)
			if not os.path.exists(melspec_Path + '/' + x):
				os.makedirs(melspec_Path +'/'+ x)

	
	
	wav_paths = []
	for directory in os.listdir(audio_path):
		for audio_file in os.listdir(audio_path+"/"+directory):
			wav_paths += [audio_path+"/"+directory+"/"+audio_file]

	#preprocess_with_mel_spectrogram(wav_paths, melspec_Path, n_mels, sample_rate_global)
	
	wav_path_1 = wav_paths[:math.floor(len(wav_paths)/4)]
	wav_path_2 = wav_paths[math.floor(len(wav_paths)/4):math.floor(len(wav_paths)/2)]
	wav_path_3 = wav_paths[math.floor(len(wav_paths)/2):math.floor(len(wav_paths)/2)+math.floor(len(wav_paths)/4)]
	wav_path_4 = wav_paths[math.floor(len(wav_paths)/2)+math.floor(len(wav_paths)/4):]
 	


	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(preprocess_with_mel_spectrogram, wav_path_1, melspec_Path, n_mels, sample_rate_global)
		path_2 = executor.submit(preprocess_with_mel_spectrogram, wav_path_2, melspec_Path, n_mels, sample_rate_global)
		path_3 = executor.submit(preprocess_with_mel_spectrogram, wav_path_3, melspec_Path, n_mels, sample_rate_global)
		path_4 = executor.submit(preprocess_with_mel_spectrogram, wav_path_4, melspec_Path, n_mels, sample_rate_global)				
		

	
	paths= []
	for directory in os.listdir(melspec_Path):
		for melspec_file in os.listdir(melspec_Path+"/"+directory):
			paths += [melspec_Path+"/"+directory+"/"+melspec_file]

	print(len(paths))
	print("Start normalization of data.")
	
	balanced_paths = data_balancing.balance_files(paths)
	with open("balanced_dataset_mel.txt", "w") as f:
		json.dump(balanced_paths,f)
	mean, std = normalization_with_mel_spectrorgram(balanced_paths, n_mels=n_mels)
	print("Processing finished.")

	

	"""
############################################################################################
##############  Preprocessing of test  set with melspectrogram ############################
############################################################################################
	
	mean = np.loadtxt("mean_mel.txt")
	std = np.loadtxt("std_mel.txt")

	n_mels = 128
	sample_rate_global = 16000
	
	test_path = "data/test/audio/"
	melsepc_Path = 'data/melspec/test'
	mfcc_Path = "data/mfcc_v5/test/"
	test_files = []

	
	if not os.path.exists(melsepc_Path):
		os.makedirs(melsepc_Path)

	if not os.path.exists(melsepc_Path):
		os.makedirs(melsepc_Path)
	
	for f in os.listdir(test_path):
		test_files.append(test_path+f)
	
	wav_path_1 = test_files[:math.floor(len(test_files)/4)]
	wav_path_2 = test_files[math.floor(len(test_files)/4):math.floor(len(test_files)/2)]
	wav_path_3 = test_files[math.floor(len(test_files)/2):math.floor(len(test_files)/2)+math.floor(len(test_files)/4)]
	wav_path_4 = test_files[math.floor(len(test_files)/2)+math.floor(len(test_files)/4):]

	print("files distributed")

	#preprocess_with_mel_spectrogram(test_files, melsepc_Path, n_mels, sample_rate_global)

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(preprocess_with_mel_spectrogram, wav_path_1, melsepc_Path, n_mels, sample_rate_global)
		path_2 = executor.submit(preprocess_with_mel_spectrogram, wav_path_2, melsepc_Path, n_mels, sample_rate_global)
		path_3 = executor.submit(preprocess_with_mel_spectrogram, wav_path_3, melsepc_Path, n_mels, sample_rate_global)
		path_4 = executor.submit(preprocess_with_mel_spectrogram, wav_path_4, melsepc_Path, n_mels, sample_rate_global)				

	print("files processed")
	mel_paths =[]
	for f in os.listdir(melsepc_Path):
		mel_paths.append(melsepc_Path+f)

	print(len(mel_paths))

	print("start normalization")
	melspec_path_1 = mel_paths[:math.floor(len(mel_paths)/4)]
	melspec_path_2 = mel_paths[math.floor(len(mel_paths)/4):math.floor(len(mel_paths)/2)]
	melspec_path_3 = mel_paths[math.floor(len(mel_paths)/2):math.floor(len(mel_paths)/2)+math.floor(len(mel_paths)/4)]
	melspec_path_4 = mel_paths[math.floor(len(mel_paths)/2)+math.floor(len(mel_paths)/4):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_1, mean, std)
		path_2 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_2, mean, std)
		path_3 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_3, mean, std)
		path_4 = executor.submit(single_normalization_with_mel_spectrogram, melspec_path_4, mean, std)	

	print("finished")

