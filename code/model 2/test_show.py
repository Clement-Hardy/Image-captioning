import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from termcolor import colored
from utils import load_image, list_to_sentence
from keras.applications.resnet50 import preprocess_input
from nltk.translate.bleu_score import sentence_bleu

def test_show_and_tell(model, name_image, path_images, real_legend=None):
    
    dir_image = os.path.join(path_images, name_image)
    image = cv2.imread(dir_image)
    image = cv2.resize(image, (224, 224))
    image_pred = preprocess_input(image)
    
    plt.figure(figsize=(15,7))
    plt.imshow(image)
    plt.show()
    sampling = list_to_sentence(model.predict_sampling(image_pred[np.newaxis,:])[0])
    if real_legend!=None:
        print(colored("Real caption:", 'red'),  "{}\n".format(real_legend))
    print(colored("Sampling:", "red")," {}\n".format(sampling))
    K = [1, 2,5,10, 20, 30]
    for k in K:
        beam = list_to_sentence(model.predict_beam_search(image[np.newaxis,:], beam_size=k)[0])
        print(colored("BeamSearch with k={}:".format(k), 'red')," {}\n".format(beam))
    

def score_bleu_sampling(model, dict_image_legend, path_images):
    score_sampling = []
    for name in dict_image_legend:
        image_pred = load_image(model_cnn=model.type_model_cnn, path_images=path_images, name_image=name)
        sampling = model.predict_sampling(image_pred[np.newaxis,:])[0]
  
        score_sampling.append(sentence_bleu(dict_image_legend[name], sampling,  weights=(1, 0, 0, 0)))

    return score_sampling


def score_bleu_beam_search(model, dict_image_legend, path_images, k=10):
    score_beam = []
    for name in dict_image_legend:
        image_pred = load_image(model_cnn=model.type_model_cnn, path_images=path_images, name_image=name)
        
        beam = model.predict_beam_search(image_pred[np.newaxis,:], beam_size=k)[0]
        score_beam.append(sentence_bleu(dict_image_legend[name], beam,  weights=(1, 0, 0, 0)))
        
    return score_beam