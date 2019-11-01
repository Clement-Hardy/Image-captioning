from keras.applications.resnet50 import preprocess_input
import keras
import numpy as np
import os
import cv2
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils import load_image

def data_generator(names_images, path_images, legends, max_length_legend, vocab_size, batch_size=2):
        current_position = 0
        
        while True:
            X = np.zeros((batch_size, 224, 224, 3))
            input_y = np.zeros((batch_size,
                                max_length_legend, vocab_size))
            output_y = np.zeros((batch_size,
                                 max_length_legend, vocab_size))
            count_in_batch = 0
            while count_in_batch<batch_size:
                if current_position== len(legends):
                    current_position = 0
                y = legends[current_position]
                y = pad_sequences([y], maxlen=max_length_legend, padding="post")
                y = to_categorical(y, num_classes=vocab_size)[0]
      
                input_y[count_in_batch, :,:] = y
                output_y[count_in_batch, :-1,:] = y[1:,:]
                output_y[count_in_batch,-1,:] = y[-1,:]
                X[count_in_batch,:] = load_image(path_images=path_images, name_image=names_images[current_position])
                count_in_batch += 1
                current_position+=1   
            yield [[X, input_y], output_y]   

"""    
def data_generator(names_images, path_images, legends, max_length_legend, vocab_size, batch_size=15):
        
        current_position = 0
        
        while True:
            X = np.empty((batch_size, 224, 224, 3))
            input_y = []
            output_y = []
            count_in_batch = 0
            while count_in_batch<batch_size:
                if current_position== len(legends):
                    current_position = 0
                y = legends[current_position]
                for i in range(len(y)):
                    input_y.append(y[:i])   
                    output_y.append(y[i])
                    X[count_in_batch,:] = load_image(path_images=path_images, name_image=names_images[current_position])
                    
                    count_in_batch += 1
                    
                    if count_in_batch==batch_size:
                        input_y = np.array(input_y)
                        output_y = np.array(output_y)
                        input_y = pad_sequences(input_y, maxlen=max_length_legend, padding="post")
                        output_y = to_categorical(output_y, num_classes=vocab_size)
                        
                        yield [[X, input_y], output_y]
                        X = np.empty((batch_size, 224, 224, 3))
                        input_y = []
                        output_y = []
                        count_in_batch = 0
                current_position+=1
"""