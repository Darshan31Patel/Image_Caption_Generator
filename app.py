import streamlit as st
from PIL import Image
import numpy as np
import os
from keras.utils import load_img,img_to_array,pad_sequences
from keras.models import load_model,Model
from keras.applications.vgg16 import VGG16,preprocess_input
import pickle

tokenizer = pickle.load(open('tokenizer.pkl','rb'))
mapping = pickle.load(open('mapping.pkl','rb'))
features = pickle.load(open('features.pkl','rb'))
model = load_model('best_model.h5')
max_length = 34

vgg_model = VGG16() 
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs,             
                  outputs=vgg_model.layers[-2].output)

st.title("Image Caption Generator")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1  # Success
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        return 0  # Failure

def feature_extraction(img_path,model):
    image = load_img(img_path,target_size=(224,224,3))
    img_array = img_to_array(image)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    feature = vgg_model.predict(preprocessed_img)
    return feature

def idx_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

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

uploaded_file = st.file_uploader("Choose a file : ")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # feature extraction
        feature = feature_extraction(os.path.join("uploads", uploaded_file.name), vgg_model)

        # generating captions
        caption = predict_caption(model,feature,tokenizer,max_length)

        st.header(caption)
    else:
        st.header("Some error occurred in file upload")