from keras.applications.resnet50 import preprocess_input
import os
import cv2
import numpy as np
import warnings
from keras.preprocessing.text import Tokenizer

def create_dataset_flickr_8k(path_legend, path_images, start_sentence, end_sentence):
    
    file = open(path_legend, 'r')
    legends = file.readlines()
    
    list_names_images, list_legends = [], []
    token = Tokenizer()
    
    for legend in legends:
        
        name_image, sentence = legend.split('\t')
        name_image = name_image.split("#")[0]
        sentence = sentence.split('\n')[0]
        
        if os.path.exists(os.path.join(path_images, name_image)):
            sentence = start_sentence + " " + sentence + " " + end_sentence
            list_names_images.append(name_image)
            list_legends.append(sentence)
            
        else :
            warnings.warn("The image doesn't {} doesn't exist, legends of this image aren't adding in dataset.".format(name_image))
            
    token.fit_on_texts(list_legends)
    list_legends_number = token.texts_to_sequences(list_legends) 
    max_length_legend = np.max([len(sentence) for sentence in list_legends_number])
     
    
    return list_names_images, list_legends_number, max_length_legend, len(token.word_index), token.word_index

def create_dataset_flickr_30k(path_legend, path_images, start_sentence, end_sentence):
    
    file = open(path_legend, 'r')
    legends = file.readlines()
    
    list_names_images, list_legends = [], []
    token = Tokenizer()
    
    for legend in legends[1:]:
        if len(legend.split('|'))==3:
            name_image, useless, sentence = legend.split('|')
        
            if os.path.exists(os.path.join(path_images, name_image)):
                sentence = start_sentence + " " + sentence + " " + end_sentence
                list_names_images.append(name_image)
                list_legends.append(sentence)
            
            else :
                warnings.warn("The image doesn't {} doesn't exist, legends of this image aren't adding in dataset.".format(name_image))
            
    token.fit_on_texts(list_legends)
    list_legends_number = token.texts_to_sequences(list_legends)
    max_length_legend = np.max([len(sentence) for sentence in list_legends_number])
     
    
    return list_names_images, list_legends_number, max_length_legend, len(token.word_index)+1, token.word_index




def create_dataset_iaprtc12(path_legend, path_images, start_sentence, end_sentence):
    
    list_names_images, list_legends = [], []
    token = Tokenizer()
    
    for folder in os.listdir(path_legend):
        if os.path.isdir(os.path.join(path_legend, folder)):
            path = os.path.join(path_legend, folder)
            for file in os.listdir(path):
                image_name = os.path.join(folder, file.split(".")[0] + ".jpg")
                if os.path.exists(os.path.join(path_images,image_name)):
                    with open(os.path.join(path, file), encoding="ISO-8859-1") as content_file:
                        content = content_file.read()
                        sentences = content.split('<DESCRIPTION>')[1].split('</DESCRIPTION>')[0].split(";")
                        for sentence in sentences:
                            sentence = start_sentence + " " + sentence + " " + end_sentence
                            list_legends.append(sentence)
                            list_names_images.append(image_name)
                                   
    token.fit_on_texts(list_legends)
    list_legends_number = token.texts_to_sequences(list_legends)
    max_length_legend = np.max([len(sentence) for sentence in list_legends_number])
     
    
    return list_names_images, list_legends_number, max_length_legend, len(token.word_index)+1, token.word_index



def list_to_sentence(list_word, end_sentence='endsentence'):
    sentence = list_word[0]
    for i in np.arange(1, len(list_word)):
        if list_word[i] == end_sentence:
            break
        sentence = sentence + ' ' + list_word[i]
    return sentence


def build_dict_image_legend(names_images, legends):
    name = names_images[0]
    dict_image_legend = {}
    
    for i in np.arange(0,len(names_images)):
        if name==names_images[i] and i>0:
            dict_image_legend[name].append(legends[i])
        else:
            name = names_images[i]
            dict_image_legend[name] = [legends[i]]
            
    return dict_image_legend       


def build_number_dict(word_dict):
    number_dict = {}
    for i in word_dict:
        number_dict[word_dict[i]] = i
    return number_dict


def number_to_word(legends, number_dict):
    legends_word = []
    for i in range(len(legends)):
        legends_word.append([])
        for j in range(len(legends[i])):
            legends_word[i].append(number_dict[legends[i][j]])
    return legends_word


def load_image(path_images, name_image, model_cnn="resnet50"):
    if model_cnn=="resnet50":
        input_shape = (224, 224)
    elif model_cnn=="InceptionV3":
        input_shape = (299, 299)
        
    dir_image = os.path.join(path_images, name_image)
    image = cv2.imread(dir_image)
    image = cv2.resize(image, input_shape)
        
    return preprocess_input(image)