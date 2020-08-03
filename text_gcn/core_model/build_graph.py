import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

"""
:return:
ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.train.index => the indices of training docs in original doc list.

All objects above must be saved using python pickle module.


"""

path = '/home/hanh/Videos/multiligualNER/text_gcn/'
dataset = ['conll2003/train.txt','conll2003/valid.txt','conll2003/test.txt']

word_embeddings_dim = 300
word_vector_map = {}
#train
f = open(path + dataset[0], 'r')
lines = f.readlines()
train = []
train_data = []
for line in lines:
    train_data = line.split(" ")[0].replace('\n', ' ')
    train.append(train_data)

train_id = []
train_ids = []
for train_name in train:
    train_id = train.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)
print(train_ids)
f.close()

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open(path + dataset[0] + '.index', 'w')
f.write(train_ids_str)
f.close()

#valid 
f = open(path + dataset[1], 'r')
lines = f.readlines()
valid = []
valid_data = []
for line in lines:
    valid_data = line.split(" ")[0].replace('\n', ' ')
    valid.append(valid_data)

valid_id = []
valid_ids = []
for valid_name in valid:
    valid_id = train.index(valid_name)
    valid_ids.append(valid_id)
random.shuffle(valid_ids)
print(valid_ids)
f.close()

valid_ids_str = '\n'.join(str(index) for index in valid_ids)
f = open(path + dataset[1] + '.index', 'w')
f.write(valid_ids_str)
f.close()

#test
f = open(path + dataset[2], 'r')
lines = f.readlines()
test = []
test_data = []
for line in lines:
    test_data = line.split(" ")[0].replace('\n', ' ')
    test.append(test_data)

test_id = []
test_ids = []
for test_name in test:
    test_id = train.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)
print(test_ids)
f.close()

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open(path + dataset[2] + '.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids +  valid_ids +test_ids
print(ids)
print(len(ids))


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
vocab_size = len(vocab) #9489

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)
f = open(path + 'conll2003_wv/aconll_vocab.txt', 'w')
f.write(vocab_str)
f.close()

# # label list
# label_set = set()
# for doc_meta in shuffle_doc_name_list:
#     temp = doc_meta.split('\t')
#     label_set.add(temp[2])
# label_list = list(label_set)

# label_list_str = '\n'.join(label_list)
# f = open('data/corpus/' + dataset + '_labels.txt', 'w')
# f.write(label_list_str)
# f.close()

# # x: feature vectors of training docs, no initial features
# # slect 90% training set
# train_size = len(train_ids)
# val_size = int(0.1 * train_size)
# real_train_size = train_size - val_size  # - int(0.5 * train_size)
# # different training rates

# real_train_doc_names = shuffle_doc_name_list[:real_train_size]
# real_train_doc_names_str = '\n'.join(real_train_doc_names)

# f = open('data/' + dataset + '.real_train.name', 'w')
# f.write(real_train_doc_names_str)
# f.close()

# row_x = []
# col_x = []
# data_x = []
# for i in range(real_train_size):
#     doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
#     doc_words = shuffle_doc_words_list[i]
#     words = doc_words.split()
#     doc_len = len(words)
#     for word in words:
#         if word in word_vector_map:
#             word_vector = word_vector_map[word]
#             # print(doc_vec)
#             # print(np.array(word_vector))
#             doc_vec = doc_vec + np.array(word_vector)

#     for j in range(word_embeddings_dim):
#         row_x.append(i)
#         col_x.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
# x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
#     real_train_size, word_embeddings_dim))

# y = []
# for i in range(real_train_size):
#     doc_meta = shuffle_doc_name_list[i]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     y.append(one_hot)
# y = np.array(y)
# print(y)

# # tx: feature vectors of test docs, no initial features
# test_size = len(test_ids)

# row_tx = []
# col_tx = []
# data_tx = []
# for i in range(test_size):
#     doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
#     doc_words = shuffle_doc_words_list[i + train_size]
#     words = doc_words.split()
#     doc_len = len(words)
#     for word in words:
#         if word in word_vector_map:
#             word_vector = word_vector_map[word]
#             doc_vec = doc_vec + np.array(word_vector)

#     for j in range(word_embeddings_dim):
#         row_tx.append(i)
#         col_tx.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
# tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
#                    shape=(test_size, word_embeddings_dim))

# ty = []
# for i in range(test_size):
#     doc_meta = shuffle_doc_name_list[i + train_size]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ty.append(one_hot)
# ty = np.array(ty)
# print(ty)

# # allx: the the feature vectors of both labeled and unlabeled training instances
# # (a superset of x)
# # unlabeled training instances -> words

# word_vectors = np.random.uniform(-0.01, 0.01,
#                                  (vocab_size, word_embeddings_dim))

# for i in range(len(vocab)):
#     word = vocab[i]
#     if word in word_vector_map:
#         vector = word_vector_map[word]
#         word_vectors[i] = vector

# row_allx = []
# col_allx = []
# data_allx = []

# for i in range(train_size):
#     doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
#     doc_words = shuffle_doc_words_list[i]
#     words = doc_words.split()
#     doc_len = len(words)
#     for word in words:
#         if word in word_vector_map:
#             word_vector = word_vector_map[word]
#             doc_vec = doc_vec + np.array(word_vector)

#     for j in range(word_embeddings_dim):
#         row_allx.append(int(i))
#         col_allx.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
# for i in range(vocab_size):
#     for j in range(word_embeddings_dim):
#         row_allx.append(int(i + train_size))
#         col_allx.append(j)
#         data_allx.append(word_vectors.item((i, j)))


# row_allx = np.array(row_allx)
# col_allx = np.array(col_allx)
# data_allx = np.array(data_allx)

# allx = sp.csr_matrix(
#     (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

# ally = []
# for i in range(train_size):
#     doc_meta = shuffle_doc_name_list[i]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ally.append(one_hot)

# for i in range(vocab_size):
#     one_hot = [0 for l in range(len(label_list))]
#     ally.append(one_hot)

# ally = np.array(ally)

# print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

# '''
# Doc word heterogeneous graph
# '''

# # word co-occurence with context windows
# window_size = 20
# windows = []

# for doc_words in shuffle_doc_words_list:
#     words = doc_words.split()
#     length = len(words)
#     if length <= window_size:
#         windows.append(words)
#     else:
#         # print(length, length - window_size + 1)
#         for j in range(length - window_size + 1):
#             window = words[j: j + window_size]
#             windows.append(window)
#             # print(window)


# word_window_freq = {}
# for window in windows:
#     appeared = set()
#     for i in range(len(window)):
#         if window[i] in appeared:
#             continue
#         if window[i] in word_window_freq:
#             word_window_freq[window[i]] += 1
#         else:
#             word_window_freq[window[i]] = 1
#         appeared.add(window[i])

# word_pair_count = {}
# for window in windows:
#     for i in range(1, len(window)):
#         for j in range(0, i):
#             word_i = window[i]
#             word_i_id = word_id_map[word_i]
#             word_j = window[j]
#             word_j_id = word_id_map[word_j]
#             if word_i_id == word_j_id:
#                 continue
#             word_pair_str = str(word_i_id) + ',' + str(word_j_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1
#             # two orders
#             word_pair_str = str(word_j_id) + ',' + str(word_i_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1

# row = []
# col = []
# weight = []

# # pmi as weights

# num_window = len(windows)

# for key in word_pair_count:
#     temp = key.split(',')
#     i = int(temp[0])
#     j = int(temp[1])
#     count = word_pair_count[key]
#     word_freq_i = word_window_freq[vocab[i]]
#     word_freq_j = word_window_freq[vocab[j]]
#     pmi = log((1.0 * count / num_window) /
#               (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
#     if pmi <= 0:
#         continue
#     row.append(train_size + i)
#     col.append(train_size + j)
#     weight.append(pmi)

# # word vector cosine similarity as weights

# '''
# for i in range(vocab_size):
#     for j in range(vocab_size):
#         if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
#             vector_i = np.array(word_vector_map[vocab[i]])
#             vector_j = np.array(word_vector_map[vocab[j]])
#             similarity = 1.0 - cosine(vector_i, vector_j)
#             if similarity > 0.9:
#                 print(vocab[i], vocab[j], similarity)
#                 row.append(train_size + i)
#                 col.append(train_size + j)
#                 weight.append(similarity)
# '''

# # doc word frequency
# doc_word_freq = {}

# for doc_id in range(len(shuffle_doc_words_list)):
#     doc_words = shuffle_doc_words_list[doc_id]
#     words = doc_words.split()
#     for word in words:
#         word_id = word_id_map[word]
#         doc_word_str = str(doc_id) + ',' + str(word_id)
#         if doc_word_str in doc_word_freq:
#             doc_word_freq[doc_word_str] += 1
#         else:
#             doc_word_freq[doc_word_str] = 1

# for i in range(len(shuffle_doc_words_list)):
#     doc_words = shuffle_doc_words_list[i]
#     words = doc_words.split()
#     doc_word_set = set()
#     for word in words:
#         if word in doc_word_set:
#             continue
#         j = word_id_map[word]
#         key = str(i) + ',' + str(j)
#         freq = doc_word_freq[key]
#         if i < train_size:
#             row.append(i)
#         else:
#             row.append(i + vocab_size)
#         col.append(train_size + j)
#         idf = log(1.0 * len(shuffle_doc_words_list) /
#                   word_doc_freq[vocab[j]])
#         weight.append(freq * idf)
#         doc_word_set.add(word)

# node_size = train_size + vocab_size + test_size
# adj = sp.csr_matrix(
#     (weight, (row, col)), shape=(node_size, node_size))

# # dump objects
# f = open("conll2003_wv/ind.train.x", 'wb')
# pkl.dump(x, f)
# f.close()

# f = open("conll2003_wv/ind.train_label.y", 'wb')
# pkl.dump(y, f)
# f.close()

# f = open("conll2003_wv/ind.test.tx", 'wb')
# pkl.dump(tx, f)
# f.close()

# f = open("conll2003_wv/ind.test_label.ty", 'wb')
# pkl.dump(ty, f)
# f.close()

# # f = open("conll2003_wv/ind.{}.allx".format(dataset), 'wb')
# # pkl.dump(allx, f)
# # f.close()

# # f = open("conll2003_wv/ind.{}.ally".format(dataset), 'wb')
# # pkl.dump(ally, f)
# # f.close()

# f = open("conll2003_wv/ind.adj", 'wb')
# pkl.dump(adj, f)
# f.close()
