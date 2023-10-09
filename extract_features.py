import tensorflow
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Model
from keras.utils import load_img,img_to_array
from tqdm import tqdm
import numpy as np
import os
import pickle


def create_model():
    model = VGG16()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    # print(model.summary())
    return model


def extract_img_feature_batch(img_paths,model):
    features_batch = {}
    for img_path in tqdm(img_paths):
        image = load_img(img_path,target_size=(224,224,3))
        image = img_to_array(image)
        image = np.expand_dims(image,axis=0)
        preprocessed_input = preprocess_input(image)
        feature = model.predict(preprocessed_input,verbose=0)
        image_id = (img_path.split('\\')[1]).split('.')[0]
        features_batch[image_id] = feature
    return features_batch

directory = 'Flickr8k_Dataset'
batch_size = 32
filenames = []
for file in os.listdir(directory):
    filenames.append(os.path.join(directory, file))
total_images = len(filenames)
num_batches = (total_images + batch_size - 1) // batch_size

model = create_model()
features = {}

for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_images)
    batch_filenames = filenames[start_idx:end_idx]
    batch_features = extract_img_feature_batch(batch_filenames, model)
    features.update(batch_features)

with open('features.pkl','wb') as f:
    pickle.dump(features,f)


