import numpy as np
import random
import data_wrangler
import math
import os
import sys
from scipy.io import wavfile

import concurrent.futures


noise_dir 		= "data/train/audio/_background_noise_/"
sampling_rate 	= 16000
noise_list 		= [noise_dir+fn for fn in os.listdir(noise_dir)]

def measuereEnergyPower(data):
	"""
	Measures the energery power of a audio file

	Args:
	data:		 	audio file

	Returns:
	energy_level: 	dB energy leve of audio signal	

	"""

	max_amplitude = np.absolute(np.iinfo(data.dtype).max)
	energy_level = np.absolute(np.max(data))
	energy_level = 20*math.log10(energy_level/max_amplitude)
	return energy_level

def add_noise(data, noise, SNR):
	"""
	Adds noise to a signal at a given SNR

	Args:
	data:		 	audio file
	noise: 			noise to be added to clean signal
	SNR 			signal to noise ratio

	Returns:
	noisy_data: 	signal mixed with noise

	"""

	dB_signal		= measuereEnergyPower(data)
	dB_noise  		= measuereEnergyPower(noise)

	try:
		scale_factor =  (dB_signal /  dB_noise) / 10**(SNR/10)
	except:
		scale_factor = 0.005

	noisy_data = data + scale_factor* noise

	return noisy_data

def choose_random_noise(data, noise_data_list):
	"""
	Chooses random noise out of a list

	Args:
	data:		 		audio file
	noise_data_list: 	list of all noise types

	Returns:
	noise: 				cut out of random noise with same length as data

	"""
	rand_noise 	= random.randint(0,len(noise_data_list)-1)

	noise_data 	= noise_data_list[rand_noise]
	rand_start = random.randint(0, len(noise_data)-len(data))

	noise = noise_data[rand_start:rand_start+len(data)]
	return noise

def apply_noise_to_p(paths, percentage, noise_data_list, process):
	"""
	Adds noise with probability p.

	Args:
	paths:		 		path to all audio files
	percentage:			with which percentage should noise be added
	noise_data_list: 	list of all noise types
	process:			identifier for visualization

	Returns:
	noise: 				cut out of random noise with same length as data

	"""
	iteration = 1
	for d in paths:
		print("%s: Adding noise for class %d of %d." % (process, iteration,  len(paths)))
		for file in os.listdir(d):

			rand = random.random()
			if (rand > percentage) and ("_noisy" not in file) and (".wav" in file):
				signal_rate, signal = data_wrangler.loadAudio(d+"/"+file)
				noise = choose_random_noise(signal, noise_data_list)
				noisy_signal = add_noise(signal, noise, 4)
				noisy_signal = np.asarray(noisy_signal, dtype=np.int16)
				new_filename = d+"/"+file.split(".")[0]+"_noisy.wav"
			
				wavfile.write(new_filename,signal_rate, noisy_signal)
		iteration +=1






if __name__ == "__main__":
	noise_data_list = []
	for n in noise_list:
		if n.endswith(".wav"):
			sr, s = data_wrangler.loadAudio(n)
			noise_data_list.append(s)


	paths = []
	audio_path = "data/train/audio/"
	for d in os.listdir(audio_path):
		if d != "_background_noise_":
			paths.append(audio_path+d)

	d1 = paths[0:math.ceil(len(paths)/4)]
	d2 = paths[math.ceil(len(paths)/4):math.ceil(len(paths)/2)]
	d3 = paths[math.ceil(len(paths)/2):math.ceil(len(paths)/4)+math.ceil(len(paths)/2)]
	d4 = paths[math.ceil(len(paths)/4)+math.ceil(len(paths)/2):]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		path_1 = executor.submit(apply_noise_to_p, d1, 0.5,noise_data_list,1)
		path_2 = executor.submit(apply_noise_to_p, d2, 0.5,noise_data_list,2)
		path_3 = executor.submit(apply_noise_to_p, d3, 0.5,noise_data_list,3 )
		path_4 = executor.submit(apply_noise_to_p, d4, 0.5,noise_data_list,4)		

	

	
		