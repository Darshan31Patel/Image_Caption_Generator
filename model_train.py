import pickle
import numpy as np
from keras.utils import to_categorical,pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import Model
from keras.utils.vis_utils import plot_model

mapping = pickle.load(open('code/mapping.pkl','rb'))
features = pickle.load(open('code/features.pkl','rb'))
tokenizer = pickle.load(open('code/tokenizer.pkl','rb'))

vocab_size = len(tokenizer.word_index)+1
# print("\nTotal words:", vocab_size)
max_length = 34
# train test split
image_ids = list(mapping.keys())
split = int(len(image_ids)*0.9)
train = image_ids[0:split]
test = image_ids[split:0]

def data_generator(data_keys,mapping,features,tokenizer,max_length,vocab_size,batch_size):
    X1,X2,y = list(),list(),list()
    n=0
    while 1:
        for key in data_keys:
            n+=1
            captions = mapping[key] # it will contain list of captions i.e. 5 caption
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]

                for i in range(1,len(seq)):
                    in_seq,out_seq = seq[:i],seq[i]
                    in_seq = pad_sequences([in_seq],maxlen = max_length)[0]
                    out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0


def model():
    # image feature
    input1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(input1)
    fe2 = Dense(256, activation='relu')(fe1)
    # text feature
    input2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = model()
# plot_model(model,show_shapes=True)


epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save('code/best_model.h5')