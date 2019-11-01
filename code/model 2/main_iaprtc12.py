#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:29:11 2019

@author: hardy
"""
from utils import create_dataset_iaprtc12, build_number_dict, build_dict_image_legend, number_to_word
from keras import optimizers
from show_and_tell import Neural_model
from test_show import score_bleu_beam_search, score_bleu_sampling
import numpy as np
import matplotlib.pyplot as plt

class config(object):
    
    learning_rate = 1e-3
    loss='categorical_crossentropy'
    optimizer=optimizers.Adam
    metrics=['accuracy']
    model_cnn="resnet50"
    embedding_dim = 512
    hidden_size_lstm = 512
    dropout = 0.2
    

path_legends = '../../iaprtc12/annotations_complete_eng'
path_images = "../../iaprtc12//images"

start_sentence, end_sentence = 'beginsentence', 'endsentence'


name_images, legends, max_length_legend, vocab_size, word_dict = create_dataset_iaprtc12(path_legend=path_legends,
                                                                                path_images=path_images,
                                                                                start_sentence=start_sentence,
                                                                                end_sentence=end_sentence)
nb_train_samples = int(np.floor(len(name_images) * 0.8))
nb_val_samples = int(np.floor(len(name_images) *  0.1))
nb_test_samples = int(np.floor(len(name_images) *  0.1))




config = config

model = Neural_model(config=config, vocab_size=vocab_size, max_length_legend=max_length_legend,
                     word_dict=word_dict, pretrained_weigths_cnn="imagenet", start_sentence=start_sentence,
                     end_sentence=end_sentence, name_dataset="iaprtc12",
                     pretrained_weigths_rnn="save_best_iaprtc12.h5")

model.build_data(name_images=name_images[:nb_train_samples],
                 legends=legends[:nb_train_samples], path_images=path_images,
                 already_build=True)
model.build_data(name_images=name_images[nb_train_samples:(nb_train_samples+nb_val_samples)],
                legends=legends[nb_train_samples:(nb_train_samples+nb_val_samples)],
                path_images=path_images, already_build=True,type_data="test")

model.compile_model()
model.fit(steps_per_epoch=8, epoch=1, batch_size=100, model_check_point=False)


test_images = name_images[(nb_train_samples+nb_val_samples):]
test_legends = legends[(nb_train_samples+nb_val_samples):]

number_dict = build_number_dict(word_dict=word_dict)
dict_image = build_dict_image_legend(names_images=test_images,
                                     legends=number_to_word(test_legends,
                                                            number_dict=number_dict))

path_desktop = "C:/Users/Hard Med lenovo/Desktop"

