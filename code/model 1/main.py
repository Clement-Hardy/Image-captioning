from utils import create_dataset
from keras import optimizers
from show_and_tell import Neural_model
from datagenerator import data_generator
import numpy as np


class config(object):
    
    learning_rate = 1e-4
    loss='categorical_crossentropy'
    optimizer=optimizers.Adam
    metrics=['accuracy']
    model_cnn="resnet50"
    embedding_dim = 512
    hidden_size_lstm = 512
    dropout = 0.2
    batch_size = 2
    

path_legends = '../../flickr_8k/Flickr8k_text/Flickr8k.token.txt'
path_images = "../../flickr_8k//Flickr8k_Dataset"

start_sentence, end_sentence = 'beginsentence', 'endsentence'


name_images, legends, max_length_legend, vocab_size, word_dict = create_dataset(path_legend=path_legends,
                                                                                path_images=path_images,
                                                                                start_sentence=start_sentence,
                                                                                end_sentence=end_sentence)

nb_train_samples = int(np.floor(len(name_images) * 0.8))
nb_val_samples = int(np.floor(len(name_images) *  0.1))
nb_test_samples = int(np.floor(len(name_images) *  0.1))                                                                                

nb_train_samples = 5
nb_val_samples = 2
nb_test_samples = 2

config = config



train_generator = data_generator(names_images=name_images[:nb_train_samples], path_images=path_images,
                                 legends=legends[:nb_train_samples], vocab_size=vocab_size,
                                 max_length_legend=max_length_legend, batch_size=config.batch_size)

test_generator = data_generator(names_images=name_images[nb_train_samples:(nb_train_samples+nb_val_samples)],
                                path_images=path_images, legends=legends[nb_train_samples:(nb_train_samples+nb_val_samples)],
                                vocab_size=vocab_size, max_length_legend=max_length_legend, batch_size=config.batch_size)

model = Neural_model(config=config, vocab_size=vocab_size,
                     max_length_legend=max_length_legend, word_dict=word_dict,
                     start_sentence=start_sentence, end_sentence=end_sentence,
                     name_dataset="flickr_8k",
                     pretrained_weigths_cnn="imagenet")

model.change_tuning("rnn")
model.fit(train_generator=train_generator, test_generator=test_generator, steps_per_epoch=2, epoch=1, model_check_point=True)


#model.change_tuning("all")
#model.fit(train_generator=train_generator, test_generator=test_generator, steps_per_epoch=500, epoch=1000, model_check_point=True)