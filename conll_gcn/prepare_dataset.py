from __future__ import print_function
from utils import *

import _pickle as pkl
import argparse
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="conll2003",
                help="Dataset string ('conll2003')")
ap.add_argument("-e", "--embeddings", type=str, default="wiki_extvec",  # default="komninos_english_embeddings"
                help="Name of embeddings file in embeddings/, without .gz extension.")
ap.add_argument("-w", "--words", type=int, default=-1,
                help="Maximum number of words in the embeddings.")
ap.add_argument("-c", "--case", type=bool, default=False,
                help="If the embeddings are case sensitive.")
args = vars(ap.parse_args())

print(args)

# Define parameters
DATASET = args['dataset']
EMBEDDINGS = args['embeddings']
MAX_NUM_WORDS = args['words']
if MAX_NUM_WORDS < 0:
    MAX_NUM_WORDS = None
CASE_SENSITIVE = args['case']

embeddings_str = "embeddings/" + EMBEDDINGS + ".gz"
word2idx = word2idx_from_embeddings(
    embeddings_str, max_num_words=MAX_NUM_WORDS)

graph_preprocessor = GraphPreprocessor(
    word2idx=word2idx, case_sensitive=CASE_SENSITIVE)
graph_preprocessor.add_split('data/' + DATASET + '/train.txt', name='train')
graph_preprocessor.add_split('data/' + DATASET + '/val.txt', name='val')
graph_preprocessor.add_split('data/' + DATASET + '/test.txt', name='test')

A = graph_preprocessor.adjacency_matrices()

X = graph_preprocessor.input_data()
Y = graph_preprocessor.output_data()

word2idx = graph_preprocessor.word2idx
idx2word = {v: k for k, v in word2idx.items()}

label2idx = graph_preprocessor.label2idx
idx2label = {v: k for k, v in label2idx.items()}

meta = {'word2idx': word2idx, 'idx2word': idx2word,
        'label2idx': label2idx, 'idx2label': idx2label}

pkl.dump((A, X, Y, meta), open('pkl/' + DATASET + '.pkl', 'wb'), protocol=-1)

embedding_matrix = matrix_from_embeddings(embeddings_str, word2idx)

pkl.dump(embedding_matrix, open('pkl/' + DATASET +
                                '.embedding_matrix.pkl', 'wb'), protocol=-1)
