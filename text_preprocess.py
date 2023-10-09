import os
from tqdm import tqdm
import pickle
from keras.preprocessing.text import Tokenizer

path = 'Flickr8k_text/Flickr8k.token.txt'

def read_caption(path):
    with open(path,'r') as f:
        captions = f.read()
    return captions

captions = read_caption(path)

# create mapping for images and captions
# each image has 5 captions which are stored in list
# mapping = {'image_id' : 'list of captions'}

def mapping_caption(captions):
    mapping = {}
    for line in tqdm(captions.split('\n')):
        tokens = line.split('\t')
        if len(tokens)<2:
            continue
        image_id,caption = tokens[0],tokens[1]
        image_id = image_id.split('.')[0]
        captions = ' '.join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

mapping = mapping_caption(captions)
print('\nLength of mapping : ',len(mapping))
print('\nMapping before cleaning : ')
print(mapping['1000268201_693b08cb0e'])


def clean_mapping(mapping):
    for key,captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    return mapping

mapping = clean_mapping(mapping)
print('\nMapping after cleaning : ')
print(mapping['1000268201_693b08cb0e'])

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
print('Length of all captions : ',len(all_captions))

with open('code/mapping.pkl','wb') as f:
    pickle.dump(mapping,f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index)+1
print("\nTotal words:", vocab_size)

max_length = max(len(caption.split()) for caption in all_captions)
print('\nMax length of caption : ',max_length)

with open('code/tokenizer.pkl','wb') as f:
    pickle.dump(tokenizer,f)