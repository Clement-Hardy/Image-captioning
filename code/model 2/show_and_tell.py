import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Dropout, LSTM, Dense, Input, concatenate, TimeDistributed, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
from utils import load_image

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
        self.batch_size = 50
        self.number_dict = dict()
    
        for w in self.word_dict:
            self.number_dict[self.word_dict[w]] = w
            
        self.build_model_cnn()
        self.build_model_rnn()
        
        
    def __data_generator__(self, type_generator="train"):
        
        current_position = 0
        
        while True:
            X = np.zeros((self.batch_size, self.max_length_legend, 2048))
            input_y = np.zeros((self.batch_size,
                                self.max_length_legend, self.vocab_size))
            output_y = np.zeros((self.batch_size,
                                 self.max_length_legend, self.vocab_size))
            count_in_batch = 0
            while count_in_batch<self.batch_size:
                if type_generator=="train" and current_position== len(self.legends_train):
                    current_position = 0
                elif type_generator=="test" and current_position== len(self.legends_test):
                    current_position = 0
                if type_generator=="train":
                    y = self.legends_train[current_position]
                elif type_generator=="test":
                    y = self.legends_test[current_position]
                y = pad_sequences([y], maxlen=self.max_length_legend, padding="post")
                y = to_categorical(y, num_classes=self.vocab_size)[0]
      
                input_y[count_in_batch, :,:] = y
                output_y[count_in_batch, :-1,:] = y[1:,:]
                output_y[count_in_batch,-1,:] = y[-1,:]
                if type_generator=="train":
                    X[count_in_batch,0,:] = self.data_image_train[self.dict_legend_image_train[current_position]]
                if type_generator=="test":
                    X[count_in_batch,0,:] = self.data_image_test[self.dict_legend_image_test[current_position]]
                    
                count_in_batch += 1
                current_position+=1   
            yield [[X, input_y], output_y]
                
                              

    def __load_image(self, name_image, path_images):
        return load_image(model_cnn=self.type_model_cnn, path_images=path_images, name_image=name_image)
    
    
    def __build_dict_image(self, name_images, type_dict="train"):
        if type_dict=="train":
            self.dict_legend_image_train = {}
            size = len(self.legends_train)
        elif type_dict=="test":
            self.dict_legend_image_test = {}
            size = len(self.legends_test)
        idx_image = 0
        name = name_images[0]
        
        for i in range(size):
            if type_dict=="train":
                self.dict_legend_image_train[i] = idx_image
            elif type_dict=="test":
                self.dict_legend_image_test[i] = idx_image
            if name!=name_images[i]:
                idx_image += 1
                name = name_images[i]
    
    def build_data(self, name_images, legends, already_build=False, path_images=None, type_data="train"):
        if type_data=="train":
            self.legends_train = legends
            self.__build_dict_image(name_images=name_images, type_dict="train")
        elif type_data=="test":
            self.legends_test = legends
            self.__build_dict_image(name_images=name_images, type_dict="test")
        
        if not already_build:
            data = self.__run_model_cnn(name_images=name_images, path_images=path_images)
            
            if type_data=="train":
                np.save("result_cnn_train_{}.npy".format(self.name_dataset), data)
                self.data_image_train = data
            elif type_data=="test":
                np.save("result_cnn_test_{}.npy".format(self.name_dataset), data)
                self.data_image_test = data
        else:
            if type_data=="train":
                self.data_image_train = np.load("result_cnn_train_{}.npy".format(self.name_dataset))
            elif type_data=="test":
                self.data_image_test = np.load("result_cnn_test_{}.npy".format(self.name_dataset))
    
    def __run_model_cnn(self, name_images, path_images):
        data_image = []
        first = True
        current_name = name_images[0]
        for name in tqdm(name_images):
            if current_name!=name or first:
                image = self.__load_image(name_image=name, path_images=path_images)[np.newaxis,:]
                pred = self.model_cnn.predict(image)
                data_image.append(pred)
                current_name = name
                first = False
        return data_image
        
        
    def build_model_cnn(self):
        
        if self.type_model_cnn == "resnet50":
            self.model_cnn = ResNet50(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            
        elif self.type_model_cnn == "InceptionV3":
            self.model_cnn = InceptionV3(include_top=False, pooling="avg", weights=self.pretrained_weigths_cnn)
            

        for i in range(len(self.model_cnn.layers)):
            self.model_cnn.layers[i].name = 'cnn_{}'.format(i)
            
    def build_model_rnn(self):
        input_rnn = Input(shape=(self.max_length_legend,2048,), name="rnn_1")
        x = TimeDistributed(Dense(self.config.embedding_dim, name="rnn_2"))(input_rnn)
        x = Dropout(self.config.dropout)(x)
    
        inputs_y = Input(shape=(self.max_length_legend,self.vocab_size,), name="rnn_3")
        #x1 = Embedding(self.vocab_size, self.config.embedding_dim, name="rnn_4")(inputs_y)
        x1 = TimeDistributed(Dense(self.config.embedding_dim, name="rnn_2"))(inputs_y)
        x1 = Dropout(self.config.dropout)(x1)
        
        y1 = concatenate([x, x1], name="rnn_8")
        y1 = LSTM(self.config.hidden_size_lstm, return_sequences=True, name="rnn_9")(y1)
        outputs = TimeDistributed(Dense(self.vocab_size, activation='softmax', name="rnn_11"))(y1)
    
        self.model = Model(inputs=[input_rnn, inputs_y], outputs=outputs)
                
    
    def compile_model(self, loss=None, optimizer=None, metrics=None):
        if loss==None:
            loss = self.config.loss
        if optimizer==None:
            optimizer=self.config.optimizer(lr=self.config.learning_rate)
        if metrics==None:
            metrics = self.config.metrics
            
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        
    def __create_callback(self):
        callbacks = []
        callbacks.append(
        ModelCheckpoint('./save_best_{}.h5'.format(self.name_dataset),
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True,
                                mode='auto', period=1))

       # callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
       #                                    patience=5, verbose=1))
        
        return callbacks
    
        
    def fit(self, epoch=1, verbose=1, steps_per_epoch=3, learning_rate=1e-2, batch_size=2,
            model_check_point=False, callbacks=None):
        
        if self.pretrained_weigths_rnn!=None:
            self.model.load_weights(self.pretrained_weigths_rnn)
            
        if model_check_point and callbacks==None:
            callbacks = self.__create_callback()
        train_generator = self.__data_generator__()
        test_generator = self.__data_generator__(type_generator="test")
        self.batch_size = batch_size
        self.model.fit_generator(train_generator, epochs=epoch, verbose=verbose,
                                 steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=test_generator, validation_steps=200)
              
    
    def predict_sampling(self, images):
        
        if isinstance(images, list):
            images = np.array(images)
            
        if images.ndim==3:
            images = images[np.newaxis,:]
        sentences = []
        
        for i in range(images.shape[0]):
            image = np.zeros((1,self.max_length_legend,2048))
            input_y = np.zeros((1,self.max_length_legend,self.vocab_size))
            
            image[0,0,:] = self.model_cnn.predict(images[i][np.newaxis,:])
            sentence = [self.start_sentence]
            input_y[0,0,self.word_dict[self.start_sentence]] = 1
    
            
            for j in range(self.max_length_legend-1):
                pred_proba = self.model.predict([image, input_y])[0,j,:]
                index_best = np.argmax(pred_proba)
                if index_best==0:
                    index_best+=1
                word_argmax = self.number_dict[index_best]
                sentence.append(word_argmax)
                coord = j+1
                input_y[0,coord,index_best] = 1
            
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
            image = np.zeros((1,self.max_length_legend,2048))
            image[0,0,:] = self.model_cnn.predict(images[i][np.newaxis,:])
            
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
                        if best==0:
                            best+=1
                        current_proba = proba_y
                        current_sentence[0,j+1,best] = 1
                        current_proba *= pred_proba[best]
                        temp_proba.append(current_proba)
                        temp_input.append(current_sentence.copy())
                        current_sentence[0,j+1,best] = 0
                index_news = np.argsort(-np.array(temp_proba))[:beam_size]
                list_input = [temp_input[index] for index in index_news]
                list_proba = [temp_proba[index] for index in index_news]
            final_sentence = np.argmax(list_input[np.argmax(list_proba)].copy()[0], axis=1)
            sentences.append(final_sentence[1:])
        results = []
        for j in range(len(sentences)):
            sentence = sentences[j]
            results.append([])
            for i in range(len(sentence)):
                results.append(self.number_dict[sentence[i]])
        return results
        
    