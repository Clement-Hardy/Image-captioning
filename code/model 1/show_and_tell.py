import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
from keras.layers import Embedding, Dropout, Lambda, Reshape, LSTM, Dense, Input, concatenate, TimeDistributed, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

class Neural_model:
    
    def __init__(self, config, vocab_size, max_length_legend, word_dict,
                 start_sentence, end_sentence, name_dataset, pretrained_weigths_cnn=None,
                 pretrained_weigths_rnn=None):
        
        self.max_length_legend = max_length_legend
        self.vocab_size = vocab_size
        self.start_sentence = start_sentence
        self.end_sentence = end_sentence
        self.name_dataset = name_dataset
        
        self.config = config
        self.type_model_cnn = config.model_cnn
        self.pretrained_weigths_cnn = pretrained_weigths_cnn
        self.pretrained_weigths_rnn = pretrained_weigths_rnn
        self.is_cnn_layer = [],
        self.word_dict = word_dict
        self.batch_size = self.config.batch_size
        
        self.number_dict = {}
        for w in self.word_dict:
            self.number_dict[self.word_dict[w]] = w
        
        self.build_model()
        
    def build_model(self):
        
        if self.type_model_cnn == "resnet50":
            model_cnn = ResNet50(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            
        elif self.type_model_cnn == "VGG16":
            model_cnn = VGG16(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            
        elif self.type_model_cnn == "InceptionV3":
            model_cnn = InceptionV3(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            
        elif self.type_model_cnn == "MobileNetV2":
            model_cnn = MobileNetV2(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            
        
        for i in range(len(model_cnn.layers)):
            model_cnn.layers[i].name = 'cnn_{}'.format(i)  
            
        constant = K.variable(np.zeros((1, self.max_length_legend-1, 2048)))

        constant = K.repeat_elements(constant,rep=self.batch_size,axis=0)
        
        out = Reshape(target_shape=(1,2048))(model_cnn.output)
        input_rnn = Lambda(lambda x: K.concatenate([x,constant],axis=1),
                           output_shape=(self.max_length_legend, 2048))(out)
        
        x = TimeDistributed(Dense(self.config.embedding_dim, name="rnn_2"))(input_rnn)
        x = Dropout(self.config.dropout, name="dropout_x")(x)
    
        inputs_y = Input(shape=(self.max_length_legend,self.vocab_size,), name="rnn_3")
        #x1 = Embedding(self.vocab_size, self.config.embedding_dim, name="rnn_4")(inputs_y)
        x1 = TimeDistributed(Dense(self.config.embedding_dim, name="rnn_2"))(inputs_y)
        x1 = Dropout(self.config.dropout, name="dropout_x1")(x1)
        
        y1 = concatenate([x, x1], name="rnn_8")
        y1 = LSTM(self.config.hidden_size_lstm, return_sequences=True, name="rnn_9")(y1)
        outputs = TimeDistributed(Dense(self.vocab_size, activation='softmax', name="rnn_11"))(y1)
    
    
        self.model = Model(inputs=[model_cnn.input, inputs_y], outputs=outputs)
        
        
    def __create_callback(self):
        callbacks = []
        callbacks.append(
        ModelCheckpoint('./save_best_{}.h5'.format(self.name_dataset),
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True,
                                mode='auto', period=1))
        
        return callbacks
    
    def compile_model(self, loss=None, optimizer=None, metrics=None):
        if loss==None:
            loss = self.config.loss
        if optimizer==None:
            optimizer=self.config.optimizer(lr=self.config.learning_rate)
        if metrics==None:
            metrics = self.config.metrics
            
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        
    def fit(self, train_generator, test_generator, epoch=1, verbose=1, steps_per_epoch=3, model_check_point=False, callbacks=None):
        
        if model_check_point and callbacks==None:
            callbacks = self.__create_callback()
        self.model.fit_generator(train_generator, epochs=epoch,
                                 verbose=verbose, steps_per_epoch=steps_per_epoch,
                                 callbacks=callbacks, validation_data=test_generator, validation_steps=1)
        
    def change_tuning(self, tune="all"):
        
        for i in range(len(self.model.layers)):
            if tune=="cnn":
                if "cnn" in self.model.layers[i].name:
                    self.model.layers[i].trainable = True
                else : 
                    self.model.layers[i].trainable = False
                
            elif tune=="rnn":
                if "rnn" in self.model.layers[i].name:
                    self.model.layers[i].trainable = True
                else : 
                    self.model.layers[i].trainable = False
                
            elif tune=="all":
                self.model.layers[i].trainable = True
                
        self.compile_model()
        
        
    
    def predict_sampling(self, images):
        
        if isinstance(images, list):
            images = np.array(images)
            
        if images.ndim==3:
            images = images[np.newaxis,:]
        sentences = []
        
        for i in range(images.shape[0]):
            img = np.zeros((self.batch_size,224,224,3))
            img[0,:] = images[i]
            input_y = np.zeros((self.batch_size,self.max_length_legend,self.vocab_size))
            
            sentence = [self.start_sentence]
            input_y[0,0,self.word_dict[self.start_sentence]] = 1
    
            
            for j in range(self.max_length_legend-1):
                pred_proba = self.model.predict([img, input_y])[0,j,:]
                index_best = np.argmax(pred_proba)

                word_argmax = self.number_dict[index_best+1]
                sentence.append(word_argmax)
                input_y[0,j+1,index_best] = 1
            
                if word_argmax==self.end_sentence:
                    break
                    
            sentences.append(sentence[1:])
        
        return sentences
    
    
    
    def predict_beam_search(self, images, beam_size=5):
        
        if isinstance(images, list):
            images = np.array(images)
            
        if images.ndim==3:
            images = images[np.newaxis,:]
        sentences = []
        
        for i in range(images.shape[0]):
            image = images[i][np.newaxis,:]
            sentence = [self.start_sentence]
            
            input_y = np.zeros((1,self.max_length_legend,self.vocab_size))
            input_y[0,0,self.word_dict[self.start_sentence]] = 1
            list_input = [input_y]
            list_proba = [1.]
            
            for j in range(self.max_length_legend-1):
                temp_proba =[]
                temp_input = []
                for input_y, proba_y in zip(list_input, list_proba):
                    
                    pred_proba = self.model.predict([image, input_y])[0,j,:]
                    index_bests = np.argsort(-pred_proba)[:beam_size]
                    current_sentence = input_y.copy()
                    
                    for best in index_bests:
                        current_proba = proba_y
                        current_sentence[0,j+1,best] = 1
                        current_proba *= pred_proba[best]
                        temp_proba.append(current_proba)
                        temp_input.append(current_sentence.copy())
                        current_sentence[0,j+1,best] = 0
                index_news = np.argsort(-np.array(temp_proba))[:beam_size]
                list_input = [temp_input[index] for index in index_news]
                list_proba = [temp_proba[index] for index in index_news]
            final_sentence = np.argmax(list_input[np.argmax(list_proba)].copy()[0], axis=1) + 1
            sentences.append(final_sentence[1:])
        results = []
        for j in range(len(sentences)):
            sentence = sentences[j]
            results.append([])
            for i in range(len(sentence)):
                results.append(self.number_dict[sentence[i]])
        return results