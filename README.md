# SimpleSpeechRecognition
## Disclaimer
The datapreprocessing pipeline is not very efficient, to improve it don't save the files locally but add noise and transform them to the wanted feature representation on the fly. </br>
The model weights are not presented here.


## Task
A simple Recurrent Neural Network is trained to distinguish audio files (length of one second). The possible classes are "noise", "down", "go", "left", "no", "off", "on", "right", "stop", "unknown", "up" and "yes".

The full task is described in the official Kaggle Competition (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data).

## Feature representation and preprocessing
The audio samples are transformed either into the *log power of the melspectrogram* or the *MFCC (from the log power of the melspectrogram)*. </br>
In both cases overlapping windows of length 20ms and an overlap of 10 ms are used.

Using a 50% chance of introducing noise at and SNR(dB) of 4 the results were improved. 

## Network
To classify the audio files into classes LSTMs combined with fully connected layers were used.</br>
E.g.:</br>

*Fully_Connected(39, activation='relu'))*</br>
*LSTM(48)* </br>
*LSTM(36)*</br>
*LSTM(36)* </br>
*LSTM(24)* </br>
*LSTM(24)* </br>
*LSTM(24)*</br>
*Fully_Connected(18)*</br>
*Fully_Connected(12, activation="softmax"))*</br>

Also have a look at the **model_zoo** file were different models are defined.


## Results
The best result with an official Kaggle Score of *72.512%* was model model_v50, which was trained with ADAM optimizer for 168 epochs with a learning rate of 0.005, and afterwards was trained with SGD for 293 epochs and a learning rate of 0.001.


* *data_augmentation*: adding noise and saving audio_files
* *data_read*: reading files, splitting them into train and test set and extracting labels, which are transformed into one hot encodings
* *data_wrangler*: transforming audio-files into MFCC or melspectrum features
* *data_balancing*: making sure each class has roughly the same size in training and validation set (i.i.d. assumption)
* *training*: training the network
* *predict_classes*: predicting the classes for a new file (for the Kaggle Competition)
* *model_zoo*: collection of different Model architectures, all based on LSTMs stacked upon each other combined with Fully-Connected Layers
* *make_plots*: Plots of Features
* *tester*: test the networks
