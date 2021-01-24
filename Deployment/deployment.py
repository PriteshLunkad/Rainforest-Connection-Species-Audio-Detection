import tensorflow as tf
import numpy as np 
import tensorflow_io as tfio
import pandas as pd
from datetime import datetime,timedelta
import os
import streamlit as st  


def final_fun_unquantized(file):

    st.write('## Running Unquantized Model')
    # Saving the starting time
    start1 = datetime.now()

    # Creating mel-spectrogram
    audio = tfio.audio.AudioIOTensor('files/'+file)
    audio_slice = audio[100:]

    # remove last dimension
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    audio_tensor = tf.cast(audio_tensor, tf.float32)

    # Convert to spectrogram
    spectrogram = tfio.experimental.audio.spectrogram(audio_tensor, nfft=2048, window=2048, stride=512)

    # Convert to mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=48000, mels=384, fmin=40, fmax=24000)

    # Convert to db scale mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)

    # Expanding the dimensions of spectrograms by 1 
    image = tf.expand_dims(mel_spectrogram, axis= -1)
    # Resizing the spectrogram
    image = tf.image.resize(image, [384, 768])
    # Converting the spectrogram to rgb
    image = tf.image.grayscale_to_rgb(image)
    # Expanding the dims for input data
    image = tf.expand_dims(image,axis = 0)

    start2 = datetime.now()
    # Creating the model
    backbone = tf.keras.applications.DenseNet121(include_top = False,input_shape = (384,768,3), weights="imagenet")

    for layer in backbone.layers[:0]:
        layer.trainable = False

    model = tf.keras.Sequential([
                backbone,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(24,bias_initializer=tf.keras.initializers.Constant(-2.))])
    
    # Loading the model weights and predicting the ouput
    model.load_weights('Models/Saved_model.h5')
    output = tf.sigmoid(model(image)).numpy()

    # Printing the time taken 
    end = datetime.now()
    st.write('Time taken to load the model and predict the value :',end - start2) 
    st.write('Time taken including preprocessing the audio files :',end - start1) 
    output = output[0].argsort()[-3:][::-1]

    st.write('Top 3 predicted species ids according to probabilities: '+str(output[0])+', '+str(output[1])+', '+str(output[2])) 

def final_fun_float16(file):
    st.write('## Running float16 quantized Model')
    # Saving the starting time
    start1 = datetime.now()

    # Creating mel-spectrogram
    audio = tfio.audio.AudioIOTensor('files/'+file)
    audio_slice = audio[100:]

    # remove last dimension
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    audio_tensor = tf.cast(audio_tensor, tf.float32)

    # Convert to spectrogram
    spectrogram = tfio.experimental.audio.spectrogram(audio_tensor, nfft=2048, window=2048, stride=512)

    # Convert to mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=48000, mels=384, fmin=40, fmax=24000)

    # Convert to db scale mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)

    # Expanding the dimensions of spectrograms by 1 
    image = tf.expand_dims(mel_spectrogram, axis= -1)
    # Resizing the spectrogram
    image = tf.image.resize(image, [384, 768])
    # Converting the spectrogram to rgb
    image = tf.image.grayscale_to_rgb(image)
    # Expanding the dims for input data
    image = tf.expand_dims(image,axis = 0)

    start2 = datetime.now()
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path='Models/float16_quantization.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pred = interpreter.get_tensor(output_details[0]['index'])
    output = tf.sigmoid(pred).numpy()

    # Printing the time taken 
    end = datetime.now()
    st.write('Time taken to load the model and predict the value :',end - start2) 
    st.write('Time taken including preprocessing the audio files :',end - start1) 
    output = output[0].argsort()[-3:][::-1]

    st.write('Top 3 predicted species ids according to probabilities: '+str(output[0])+', '+str(output[1])+', '+str(output[2])) 


def final_fun_dynamic(file):
    st.write('## Running dynamic quantized Model')
    # Saving the starting time
    start1 = datetime.now()

    # Creating mel-spectrogram
    audio = tfio.audio.AudioIOTensor('files/'+file)
    audio_slice = audio[100:]

    # remove last dimension
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    audio_tensor = tf.cast(audio_tensor, tf.float32)

    # Convert to spectrogram
    spectrogram = tfio.experimental.audio.spectrogram(audio_tensor, nfft=2048, window=2048, stride=512)

    # Convert to mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=48000, mels=384, fmin=40, fmax=24000)

    # Convert to db scale mel-spectrogram
    mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)

    # Expanding the dimensions of spectrograms by 1 
    image = tf.expand_dims(mel_spectrogram, axis= -1)
    # Resizing the spectrogram
    image = tf.image.resize(image, [384, 768])
    # Converting the spectrogram to rgb
    image = tf.image.grayscale_to_rgb(image)
    # Expanding the dims for input data
    image = tf.expand_dims(image,axis = 0)

    start2 = datetime.now()
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path='Models/dynamic_quantization.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pred = interpreter.get_tensor(output_details[0]['index'])
    output = tf.sigmoid(pred).numpy()

    # Printing the time taken 
    end = datetime.now()
    st.write('Time taken to load the model and predict the value :',end - start2) 
    st.write('Time taken including preprocessing the audio files :',end - start1) 
    output = output[0].argsort()[-3:][::-1]

    st.write('Top 3 predicted species ids according to probabilities: '+str(output[0])+', '+str(output[1])+', '+str(output[2])) 

    st.write('''Dynamic Quantized model takes very long time to execute because this quantization technique is optimized only for ARM based CPUs it is not well optimized for desktop CPUs
    	      https://github.com/tensorflow/tensorflow/issues/24665''')



st.title('Rainforest Connection Species Audio Detection')
st.write('''In this challenge we have to predict based on the sounds of various
species of birds and frogs which species the sound will belong to. Traditional methods of assessing the diversity and abundance of
species are costly and limited in space and time. So a deep learning
based approach will be very helpful to accurately detect the species in
noisy landscapes. Rainforest Connection(RFCx) created the world’s first real-time monitoring system for protecting and supporting remote systems and
unlike visual-based tracking systems like drones or satellites, RFCxrelies on acoustic sensors to monitor the ecosystem soundscapes at
different locations all year around. The system built by RFCx also has the capacity to create convolutional
neural network (CNN) models for analysis. In this problem we have to automate the detection of birds and frog species based on sound recordings. There are a total of 23 species of birds and animals present in the train
dataset so this problem is a multi-class classification problem. The link to the problem statement https://www.kaggle.com/c/rfcx-species-audio-detection ''')

st.write('## Upload a .flac audio file containing the species sound for prediction')


file = st.file_uploader('',type = ['flac'])

st.write('## Select from sample audio files')

option = st.selectbox(
        '',
        ('None','03971ab79.flac','0b3479649.flac','0d1d831a8.flac', '754568dcb.flac', '774d13e9c.flac','b16ad4d20.flac','c874f072d.flac','dcb4321a5.flac','e22487d26.flac','ef47304a3.flac'),index = 0)
executed = 0
if file != None and executed ==0:
	with open('files/'+file.name,"wb") as f: 
	    f.write(file.getbuffer())  
	audio = st.audio(file)
	final_fun_unquantized(file.name)
	final_fun_float16(file.name)
	final_fun_dynamic(file.name)
	os.remove('files/'+file.name)
	executed = 1


if option!= 'None' and executed ==0:
	st.audio('files/'+option)
	final_fun_unquantized(option)
	final_fun_float16(option)
	final_fun_dynamic(option)
	executed = 1
