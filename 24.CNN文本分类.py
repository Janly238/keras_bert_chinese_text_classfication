from keras.models import Sequential
from keras.layers import Dense,Embedding,Dropout,Conv1D,MaxPooling1D,GlobalMaxPooling1D,Flatten
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
import keras
import fasttext

VECTOR_DIR = "/home/lenovo/miao/5.word_discovery/data/sinlp/fasttext_model/finance.bin"

class CNN():
    def __init__(self,nb_words,output_dim):
        #fasttext embedding
        # fasttext_model = fasttext.load_model(VECTOR_DIR, encoding="utf-8")
        # embedding_maxtrix = np.zeros((nb_words+1,200))
        # for word,i in text_token.word_index.items():
        #     embedding_maxtrix[i]=np.asarray(fasttext_model[word])
        # embedding_layer = Embedding(nb_words+1,200,weights=[embedding_maxtrix],input_length=300,trainable=False)


        self.model=Sequential()
        self.model.add(Embedding(nb_words+1,300,input_length=300))
        # self.model.add(embedding_layer)

        #用鸿哥的一层cnn
        # self.model.add(Conv1D(64,3,activation="relu"))
        # self.model.add(GlobalMaxPooling1D())
        # self.model.add(Dense(128,activation="relu"))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(output_dim,activation="softmax"))

        #只用LSTM
        # self.model.add(LSTM(256,dropout=0.2,recurrent_dropout=0.1))

        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
        self.model.add(MaxPooling1D(3))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
              
        self.early_stop = EarlyStopping(patience=7)
        # self.model.compile(loss="sparse_categorical_crossentropy",
        #             optimizer="Adadelta",
        #             metrics=["accuracy"])
        self.model.summary()


    def model_train(self,x_train, y_train, x_test, y_test):
        self.model.fit(x_train,y_train,batch_size=128,epochs=10,validation_data=(x_test,y_test),
                        callbacks=[self.early_stop])

    def model_pred(self,x_input,batch_size):
        self.model.predict(x_input,batch_size)

if __name__ == "__main__":
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import numpy as np
    csv_file=pd.read_excel('train_test_5000_new3.xlsx')
    text=csv_file["文本"]
    text_label = csv_file["人工标注 修改"]
    y = preprocessing.LabelEncoder().fit_transform(text_label)
    y= to_categorical(y)
    text_token = Tokenizer(char_level=True)
    text_token.fit_on_texts(text)
    x=text_token.texts_to_sequences(text)
    x=pad_sequences(x,300)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    print(y.shape)
    # cnn = CNN(len(text_token.word_index.keys()),len(set(y)))
    cnn = CNN(len(text_token.word_index.keys()),y.shape[1])
    cnn.model_train(np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y))
