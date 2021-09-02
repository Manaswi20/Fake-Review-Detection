import numpy as np
import pandas as pd
from keras.layers import (Input, Embedding, Conv1D, Activation, GlobalMaxPool1D, BatchNormalization,
                          Concatenate, CuDNNLSTM, Flatten, Dropout, Dense)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical, Sequence, plot_model
import matplotlib.pyplot as plt



df = pd.read_csv("dataset.csv")

# Print first lines of `df` 
df.head()
print(df.head())

df["LABEL"] = df["LABEL"].replace(['REAL'],'0')
df["LABEL"] = df["LABEL"].replace(['FAKE'],'1')

y = df.LABEL
df.drop("LABEL", axis=1)      #where numbering of news article is done that column is dropped in dataset
X_train, X_test, y_train, y_test = train_test_split(df['REVIEW'], y, test_size=0.3, random_state=4)


X_holdout, X_test, y_holdout, y_test = train_test_split(X_test, y_test, test_size=0.66, random_state=98)
y_train, y_test, y_holdout = list(y_train), list(y_test), list(y_holdout)





tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(df['REVIEW'])
X_train = tokenizer.texts_to_sequences(X_train)
X_holdout = tokenizer.texts_to_sequences(X_holdout)
X_test = tokenizer.texts_to_sequences(X_test)



def get_embed_mat(EMBEDDING_FILE):
    embeddings_index = {}
    with open(EMBEDDING_FILE,'r',encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    all_embs = np.stack(embeddings_index.values())
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 
                                        (num_words, embed_dim))
    for word, i in word_index.items():
        if i >= num_words:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return num_words, embedding_matrix

EMBEDDING_FILE = 'glove.6B.200d.txt'
embed_dim = 200
num_words, embedding_matrix = get_embed_mat(EMBEDDING_FILE)


plot = plt.hist([len(x) for x in X_train + X_test], bins=100)
plt.savefig('results/Histogram.png')
plt.pause(5)
plt.show(block=False)
plt.close()
max_length = 50
print('Sequence length:', max_length)

#CNN - Static


num_classes = 2
layers = []
filters = [2, 3, 5]

sequence_input1 = Input(shape=(max_length, ), dtype='int32')
embedding_layer_static1 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length, trainable=False)(sequence_input1)

for sz in filters:
    conv_layer1 = Conv1D(filters=256, kernel_size=sz)(embedding_layer_static1)
    batchnorm_layer1 = BatchNormalization()(conv_layer1)
    act_layer1 = Activation('relu')(batchnorm_layer1)
    pool_layer1 = GlobalMaxPool1D()(act_layer1)
    layers.append(pool_layer1)

merged1 = Concatenate(axis=1)(layers)

drop1 = Dropout(0.5)(merged1)
dense1 = Dense(512, activation='relu')(drop1)
out1 = Dense(num_classes, activation='softmax')(dense1)

cnn_static = Model(sequence_input1, out1)
cnn_static.summary()


def top_3_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

class dataseq(Sequence):
    def __init__(self, X, y, batch_size, padding='post'):
        self.x, self.y = X, y
        self.batch_size = batch_size
        self.m = len(self.y)
        self.padding = padding

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]
        batch_y = self.y[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]

        return pad_sequences(batch_x, maxlen=max_length, truncating='post', padding=self.padding), to_categorical(
            batch_y, num_classes=num_classes)

cnn_static.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])

batch_size = 128
cnn_static_history = cnn_static.fit_generator(dataseq(X_train, y_train, batch_size), epochs=15, verbose=2,validation_data = dataseq(X_holdout, y_holdout, batch_size), shuffle=True)


plt.plot(cnn_static_history.history['acc'], label='train')
plt.plot(cnn_static_history.history['val_acc'], label='holdout')
plt.title('CNN - Static learning Accuracy curve')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/CNN - Static learning Accuracy.png')
plt.pause(5)
plt.show(block=False)
plt.close()

plt.plot(cnn_static_history.history['loss'], label='train')
plt.plot(cnn_static_history.history['val_loss'], label='holdout')
plt.title('CNN - Static learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/CNN - Static learning curve.png')
plt.pause(5)
plt.show(block=False)
plt.close()

cnn_static.evaluate_generator(dataseq(X_test, y_test, batch_size))






#CNN - Dynamic

layers = []

embedding_layer_dynamic1 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),input_length=max_length)(sequence_input1)

for sz in filters:
    conv_layer2 = Conv1D(filters=256, kernel_size=sz)(embedding_layer_dynamic1)
    batchnorm_layer2 = BatchNormalization()(conv_layer2)
    act_layer2 = Activation('relu')(batchnorm_layer2)
    pool_layer2 = GlobalMaxPool1D()(act_layer2)
    layers.append(pool_layer2)

merged2 = Concatenate(axis=1)(layers)

drop2 = Dropout(0.5)(merged2)
dense2 = Dense(512, activation='relu')(drop2)
out2 = Dense(num_classes, activation='softmax')(dense2)

cnn_dynamic = Model(sequence_input1, out2)

cnn_dynamic.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])
cnn_dynamic_history = cnn_dynamic.fit_generator(dataseq(X_train, y_train, batch_size), epochs=5, verbose=2,validation_data = dataseq(X_holdout, y_holdout, batch_size), shuffle=True)

plt.plot(cnn_dynamic_history.history['acc'], label='train')
plt.plot(cnn_dynamic_history.history['val_acc'], label='holdout')
plt.title('CNN - Dynamic learning Accuracy curve')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/CNN - Dynamic learning Accuracy curve.png')
plt.pause(5)
plt.show(block=False)
plt.close()

plt.plot(cnn_dynamic_history.history['loss'], label='train')
plt.plot(cnn_dynamic_history.history['val_loss'], label='holdout')
plt.title('CNN - Dynamic learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/CNN - Dynamic learning curve.png')
plt.pause(5)
plt.show(block=False)
plt.close()


cnn_dynamic.evaluate_generator(dataseq(X_test, y_test, batch_size))




#Ensemble
sequence_input2 = Input(shape=(max_length, ), dtype='int32')

models = [cnn_static, cnn_dynamic]

for i in range(len(models)):
    for layer in models[i].layers:
        layer.trainable = False

input_layers = [sequence_input1, sequence_input2]
output_layers = [model.output for model in models]
ensemble_merge = Concatenate()(output_layers)
ensemble_dense = Dense(100, activation='relu')(ensemble_merge)
output = Dense(num_classes, activation='softmax')(ensemble_dense)
model = Model(inputs=input_layers, outputs=output)


class meta_dataseq(Sequence):
    def __init__(self, X, y, batch_size):
        self.x, self.y = X, y
        self.batch_size = batch_size
        self.m = len(self.y)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]
        batch_y = to_categorical(self.y[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)],
                                 num_classes=num_classes)
        
        return [pad_sequences(batch_x, maxlen=max_length, truncating='post', padding='post'),
                 pad_sequences(batch_x, maxlen=max_length, truncating='post', padding='pre')], batch_y

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])
model_history = model.fit_generator(meta_dataseq(X_holdout, y_holdout, batch_size), epochs=20, verbose=2,validation_data = meta_dataseq(X_test, y_test, batch_size))


plt.plot(model_history.history['acc'], label='train')
plt.plot(model_history.history['val_acc'], label='test')
plt.title('Meta-classifer learning Accuracy curve')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/Meta-classifer learning Accuracy curve.png')
plt.pause(5)
plt.show(block=False)
plt.close()


plt.plot(model_history.history['loss'], label='train')
plt.plot(model_history.history['val_loss'], label='test')
plt.title('Meta-classifer learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('results/Meta-classifer learning curve.png')
plt.pause(5)
plt.show(block=False)
plt.close()



model_json = model.to_json()
with open('results/model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('results/model.h5')
print('Saved model to disk')