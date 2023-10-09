import pickle
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.utils import pad_sequences
from keras.models import load_model

tokenizer = pickle.load(open('code/tokenizer.pkl','rb'))
mapping = pickle.load(open('code/mapping.pkl','rb'))
features = pickle.load(open('code/features.pkl','rb'))
model = load_model('code/best_model.h5')
max_length = 34

# convert index generated to word
def idx_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

# predict caption
def predict_caption(model,image,tokenizer,max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen=max_length)
        yhat = model.predict([image,sequence],verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat,tokenizer)
        if word is None:
            break
        in_text += " " + word

        if word=='endseq':
            break
    return in_text


def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    image_path = os.path.join('Flickr8k_Dataset',image_name)
    image = Image.open(image_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model,features[image_id],tokenizer,max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)

generate_caption("1001773457_577c3a7d70.jpg")