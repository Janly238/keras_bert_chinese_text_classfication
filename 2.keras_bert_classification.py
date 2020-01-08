#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical


maxlen = 100
config_path = '/home/lenovo/miao/0.data/bert_chinese/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/lenovo/miao/0.data/bert_chinese/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/lenovo/miao/0.data/bert_chinese/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


csv_file=pd.read_excel('train_test_5000_new3.xlsx')
texts=csv_file["文本"]
text_labels = csv_file["人工标注 修改"]
y = preprocessing.LabelEncoder().fit_transform(text_labels)
nclass = len(set(y))
# y= to_categorical(y)
data=[(text,label) for text,label in zip(texts,y)]


# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 20 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 20 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(nclass, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

checkpoint = ModelCheckpoint('weights{epoch:02d}_valacc{val_acc:.4f}_valloss{val_loss:.4f}.hdf5', monitor='val_acc',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks = [checkpoint]
)
