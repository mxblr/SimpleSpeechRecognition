__author__ = "Paul Schmidt-Barbo and Maximilian Beller"

###################################################################
#################### Import required packages #####################
###################################################################

import os
from os.path import isdir, join
from pathlib import Path
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
from librosa import display
import matplotlib.pyplot as plt
import json
import data_read
import data_wrangler

###################################################################
######################## Methods ##################################
###################################################################



def plot_mfcc(path):
	"""
	Plotting the MFCC featueres of a audio file.

	Args:
	path: path to the audio file

	Returns:
	-
	"""
	mfcc = np.loadtxt(mfcc_Path + path)

	librosa.display.specshow(mfcc, x_axis='time')
	plt.colorbar()
	plt.title('MFCC - %s' % path.split("/")[0])
	plt.tight_layout()
	plt.show()

def plot_melspectogram(path):
	"""
	Plotting the melspectorgram featueres of a audio file.

	Args:
	path: path to the audio file

	Returns:
	-
	"""
	sample_rate, samples = data_wrangler.loadAudio(audio_path+path)
	if(sample_rate > 16000):
		sample_array = data_wrangler.cutAudio(samples, 16000)
		samples = sample_array[0]

	log_s = data_wrangler.melspectrogram(samples, sample_rate, number_features=128)

	librosa.display.specshow(log_s, x_axis='time')
	plt.colorbar()
	plt.title('Melspectogram - %s' % path.split("/")[0])
	plt.tight_layout()
	plt.show()


def plot_distribution(path):
	"""
	Plotting the number of files per class:

	Args:
	path: path to the audio files

	Returns:
	-
	"""
	dirs = [f for f in os.listdir(path) if isdir(join(path, f))]
	dirs.sort()
	print('Number of labels: ' + str(len(dirs)))
	
	number_of_recordings = []
	for direct in dirs:
	    waves = [f for f in os.listdir(join(path, direct)) if f.endswith('.wav')]
	    number_of_recordings.append(len(waves))

	# Plot
	fig = plt.figure()
	plt.bar(dirs, number_of_recordings,align='center')
	plt.xlabel('Classes')
	plt.ylabel('Number of audio files')
	plt.xticks(rotation=90)
	plt.show()


def plot_distribution_balanced(filename):
	"""
	Plotting the number of files per class after balancing.

	Args:
	filename: file containing all audio filess

	Returns:
	-
	"""
	with open(filename, "r") as f:
		dataset = json.load(f)

	counter = {}
	for audio_path in dataset:
		file_class = data_read.find_label(audio_path)
		if file_class not in ["yes", "no","up", "down", "left","right","on", "off","stop", "go","_background_noise_", "unknown"]:
			file_class = "unknown"
		counter[file_class] = counter.get(file_class, 0) + 1
				
	
	# Plot
	fig = plt.figure()
	plt.bar(counter.keys(), counter.values(),align='center')
	plt.xlabel('Classes')
	plt.ylabel('Number of audio files')
	plt.xticks(rotation=90)
	plt.show()

if __name__ == "__main__":
	audio_path = 'data/train/audio/'
	mfcc_Path = 'data/mfcc/train/'

	six = "six/0b09edd3_nohash_0.txt"
	on = "on/0b09edd3_nohash_0.txt"
	doing_the_dishes = "_background_noise_/doing_the_dishes.txt"
	marvin = "marvin/0a7c2a8d_nohash_0.txt"


	six_wav = "six/0b09edd3_nohash_0.wav"
	on_wav = "on/0b09edd3_nohash_0.wav"
	doing_the_dishes_wav = "_background_noise_/doing_the_dishes.wav"
	marvin_wav = "marvin/0a7c2a8d_nohash_0.wav"

	plot_mfcc(six)
	plot_mfcc(on)
	plot_mfcc(doing_the_dishes)
	plot_mfcc(marvin)

	#plot_distribution(audio_path)
	#plot_distribution_balanced("balanced_dataset.txt")

	plot_melspectogram(six_wav)
	plot_melspectogram(on_wav)
	plot_melspectogram(doing_the_dishes_wav)
	plot_melspectogram(marvin_wav)
