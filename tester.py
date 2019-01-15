import pyaudio
import numpy as np
import predict_classes
import model_zoo
import time

import cv2

CHUNK = 16000 # number of data points to read at a time
RATE = 16000 # time resolution of the recording device (Hz)

p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

# create a numpy array holding a single read of audio data
version = "v49"
weights_path = "models/%s/weights.best.hdf5" % (version)
model = model_zoo.model_v49()

while True: #to it a few times just to see
	data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
	predict_classes.predict_single_unpreprocessed(weights_path, data, model)

	k = cv2.waitKey(30) & 0xff

	if k == 27:
		break


#print(data.shape)

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()

