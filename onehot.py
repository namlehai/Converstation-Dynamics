# -*- coding: utf-8 -*-

from keras.preprocessing.text import Tokenizer

import os


abortion_dir = 'abortion'
train_dir = os.path.join(abortion_dir, 'train1')

labels = []
texts = []

for label_type in ['pro', 'against']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf-8")
            for line in f:
                texts.append(line)
                if label_type == 'against':
                    labels.append(0)
                else:
                    labels.append(1)
#            line = f.readline()
#            texts.append(f.read())
#            texts.append(line)
#            print(len(line))
#            print(texts)
            f.close()
            # if neg, label = 0; else: label = 1
#            if label_type == 'against':
#                labels.append(0)
#            else:
#                labels.append(1)
                
print("length of the texts ", len(texts))
print("length of the labels ", len(labels))

#samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer()#(num_words=1000)
#tokenizer.fit_on_texts(samples)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
one_hot_results = tokenizer.texts_to_matrix(texts, mode='binary')

print(one_hot_results.shape)
print(one_hot_results)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))