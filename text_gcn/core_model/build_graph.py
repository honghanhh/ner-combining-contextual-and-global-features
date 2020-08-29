import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from sklearn import svm
from math import log
from utils import loadWord2Vec, clean_str
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import numpy as np
import random
import os
import nltk
nltk.download('wordnet')

"""
:return:
ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object.

All objects above must be saved using python pickle module.


"""
path = '../'
dataset = ['conll2003/train.txt', 'conll2003/valid.txt', 'conll2003/test.txt']

word_embeddings_dim = 300
word_vector_map = {}

# label list
label_list = ['<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER',
              'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG']

label_list_str = '\n'.join(label_list)
f = open(path + 'conll2003_wv/conll_labels.txt', 'w')
f.write(label_list_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
f = open(path + 'conll2003_wv/conll_corpus.txt', 'r')
words = f.readlines()
for word in words[0].split():
    word_set.add(word)
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)  # 9489

vocab_str = '\n'.join(vocab)
f = open(path + 'conll2003_wv/conll_vocab.txt', 'w')
f.write(vocab_str)
f.close()

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

f = open(path + 'conll2003_wv/word_id_map.txt', 'w')
f.write(str(word_id_map))
f.close()

# train
f = open(path + dataset[0], 'r')
lines = f.readlines()
train = []
train_data = []
for line in lines:
    train_data = line.split(" ")[0].replace('\n', ' ')
    train.append(train_data)

print("Train length: ", len(train))  # 274599

train_sentence = ' '.join(str(word) for word in train)
# [22348 rows x 22348 columns]
train_sentence_dict = [x + ' .' for x in train_sentence.split(".")]

# valid
f = open(path + dataset[1], 'r')
lines = f.readlines()
valid = []
valid_data = []
for line in lines:
    valid_data = line.split(" ")[0].replace('\n', ' ')
    valid.append(valid_data)

print("Valid length: ", len(valid))  # 55045

valid_sentence = ' '.join(str(word) for word in valid)
# [8692 rows x 8692 columns]
valid_sentence_dict = [x + ' .' for x in valid_sentence.split(".")]

# test
f = open(path + dataset[2], 'r')
lines = f.readlines()
test = []
test_data = []
for line in lines:
    test_data = line.split(" ")[0].replace('\n', ' ')
    test.append(test_data)

print("Test length: ", len(test))  # 50351

test_sentence = ' '.join(str(word) for word in test)
# [8112 rows x 8112 columns]
test_sentence_dict = [x + ' .' for x in test_sentence.split(".")]


# Word definitions begin
f = open(path + 'conll2003_wv/conll_vocab.txt', 'r')
vocab = f.readlines()
f.close()

definitions = []
for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)
f = open(path + 'conll2003_wv/conll_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
# print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open(path + 'conll2003_wv/conll_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = path + 'conll2003_wv/conll_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])

# print("Word vector map", word_vector_map)
# print("Word embedding dim", word_embeddings_dim)


# Get x representation
row_x = []
col_x = []
# data_x = []
for i in range(len(train)):
    for word in train:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
    print("Word vector " + str(i), word_vector)
    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
    print("Finish iteration ", str(i))

x = sp.csr_matrix((row_x, col_x), shape=(len(train), word_embeddings_dim))
print(x)
# valid
row_vx = []
col_vx = []
for i in range(len(valid)):
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
    for j in range(word_embeddings_dim):
        row_vx.append(i)
        col_vx.append(j)
vx = sp.csr_matrix(((row_vx, col_vx)),
                   shape=(len(valid), word_embeddings_dim))

# test
row_tx = []
col_tx = []
for i in range(len(test)):
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
tx = sp.csr_matrix(((row_tx, col_tx)),
                   shape=(len(test), word_embeddings_dim))

# Co-occurence matrix
"""
def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df

x_tr = co_occurrence(train_sentence_dict, 20)
print(x_tr)
x_v = co_occurrence(valid_sentence_dict, 20)
print(x_v)
x_te = co_occurrence(test_sentence_dict, 20)
print(x_te)

# Label

# train
f = open(path + dataset[0], 'r')
lines = f.readlines()
label_train = []
label_train_data = []
for line in lines:
    if len(line.split(" ")) == 4:
        label_train_data = line.split(" ")[3].replace('\n', ' ')
        label_train.append(label_train_data)

print("Train label length: ",len(label_train))  #256145

# Adjacent matrix
"""

# dump objects
f = open(path + "conll2003_wv/conll_ind.x", 'wb')
pkl.dump(x, f)
f.close()

# f = open("data/ind.{}.y".format(dataset), 'wb')
# pkl.dump(y, f)
# f.close()

f = open(path + "conll2003_wv/conll_ind.tx", 'wb')
pkl.dump(tx, f)
f.close()

f = open(path + "conll2003_wv/conll_ind.vx", 'wb')
pkl.dump(vx, f)
f.close()
# f = open("data/ind.{}.ty".format(dataset), 'wb')
# pkl.dump(ty, f)
# f.close()

# f = open("data/ind.{}.allx".format(dataset), 'wb')
# pkl.dump(allx, f)
# f.close()

# f = open("data/ind.{}.ally".format(dataset), 'wb')
# pkl.dump(ally, f)
# f.close()

# f = open("data/ind.{}.adj".format(dataset), 'wb')
# pkl.dump(adj, f)
# f.close()
