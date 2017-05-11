# coding=utf-8
import sys

import h5py
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence

import cPickle as pickle
import thulac

from caption_reader import _read_words

EMBEDDING_DIM = 128

reload(sys)
sys.setdefaultencoding('utf8')

class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = 32
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.images_data_init()
        self.captions_data_init()

    def images_data_init(self):
        data = h5py.File('images/image_vgg19_fc2_feature_677004464.h5')
        self.images_train_set = data['train_set'].value
        self.images_validation_set = data['validation_set'].value

    def captions_data_init(self):
        with open('captions/word_to_id.txt') as f:
            text = f.read().encode('utf-8').split("\n")
            self.vocab_size = len(text)
            word_to_id = [(line.split(" "))for line in text]
            self.word_index = {}
            self.index_word = {}
            for (word,i) in word_to_id:
                self.word_index[word]=int(i)
                self.index_word[int(i)]=word
        temp = 0
        with open('captions/train.txt', 'r') as f:
            text = f.read().decode("utf-8").split("\r\n")
            for line in text:
                if not line.isdigit():
                    temp+=1
        self.total_samples = temp
        print "Vocabulary size: "+str(self.vocab_size)
        print "Maximum caption length: "+str(self.max_cap_len)
        print "Variables initialization done!"

    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print "Generating data..."
        gen_count = 0
        image_caption_pairs=[]
        with open('captions/train.txt', 'r') as f:
            text = f.read().decode("utf-8").split("\r\n")
            for line in text:
              if line.isdigit():
                image_id = int(line)
              else:
                image_caption_pairs.append((image_id, line))
        

        thu1 = thulac.thulac(seg_only=True)
        
        total_count = 0
        while 1:
            for image_id, text in image_caption_pairs:
                current_image = self.images_train_set[int(image_id)-1]
                words = _read_words(text, thu1)
                for i in range(len(words)-1):
                    total_count+=1
                    #####################################
                    #TODO: set partial and next_words
                    partial = [self.word_index[txt] for txt in words[:i+1]]
                    partial_caps.append(partial)
                    next_w = np.zeros(self.vocab_size)
                    next_w[self.word_index[words[i+1]]] = 1
                    next_words.append(next_w)
                    #####################################
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print "yielding count: "+str(gen_count)
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        #base_model.trainable=False
        image_model = Sequential()
        #image_model.add(base_model)
        #image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))

        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256,return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        print "Model created!"

        if(ret_model==True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self,index):
        return self.index_word[index]

if __name__ == '__main__':
    cg = CaptionGenerator()
    model = cg.create_model()