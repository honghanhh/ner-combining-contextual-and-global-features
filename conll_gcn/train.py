from __future__ import print_function

from keras.layers import Input, Dropout, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from layers.graph import SpectralGraphConvolution
from utils import *
import evaluation

import pickle as pkl

import time
import os
import sys
import time
import argparse

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='conll2003',
                help="Dataset string ('conll2003')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-do", "--dropout", type=float, default=0.5,
                help="Dropout rate")
ap.add_argument("-lr", "--learnrate", type=float, default=0.001,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")
ap.add_argument("-b", "--batch", type=int, default=8,
                help="Batch size")

args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args['dataset']
EPOCHS = args['epochs']
LR = args['learnrate']
L2 = args['l2norm']
DO = args['dropout']
BATCH_SIZE = args['batch']

print("Loading dataset...")

A, X, Y, meta = pkl.load(open('pkl/' + DATASET + '.pkl', 'rb'))

print("Loading embedding matrix...")

embedding_matrix = pkl.load(
    open('pkl/' + DATASET + '.embedding_matrix.pkl', 'rb'))

print("Processing dataset...")

val_y = load_output(A, X, Y, 'val')
test_y = load_output(A, X, Y, 'test')

num_nodes = A['train'][0][0].shape[0]
num_relations = len(A['train'][0]) - 1
num_labels = len(meta['label2idx'])

print("Number of nodes: {}".format(num_nodes))
print("Number of relations: {}".format(num_relations))
print("Number of classes: {}".format(num_labels))

# Define model inputs
X_in = Input(shape=(num_nodes, ))
A_in = [Input(shape=(num_nodes, num_nodes)) for _ in range(num_relations)]

print("Define model")

# Define model architecture
X_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[
                        embedding_matrix], trainable=False)(X_in)
H = SpectralGraphConvolution(256, activation='relu')([X_embedding] + A_in)
H = Dropout(DO)(H)
H = SpectralGraphConvolution(256, activation='relu')([H] + A_in)
H = Dropout(DO)(H)
output = Dense(num_labels, activation='softmax')(H)

# Compile model
model = Model(inputs=[X_in] + A_in, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR))
model.summary()

# Fit
for epoch in range(EPOCHS):

    print("=== EPOCH {} ===".format(epoch + 1))

    model.fit_generator(batch_generator(A, X, Y, 'train', batch_size=BATCH_SIZE),
                        steps_per_epoch=len(A['train'])//BATCH_SIZE, verbose=1)

    val_predictions = model.predict_generator(batch_generator(
        A, X, Y, 'val', batch_size=BATCH_SIZE), steps=len(A['val'])//BATCH_SIZE, verbose=1)

    val_predicted_labels, val_actual_labels = evaluation.predict_labels(
        val_predictions, val_y, meta['idx2label'])
    val_precision, val_recall, val_f1 = evaluation.compute_scores(
        val_predicted_labels, val_actual_labels)

    print("=== Validation Results ===")
    print("Precision: {:.2f}%".format(val_precision * 100))
    print("Recall: {:.2f}%".format(val_recall * 100))
    print("F1: {:.2f}".format(val_f1 * 100))

    test_predictions = model.predict_generator(batch_generator(
        A, X, Y, 'test', batch_size=8), steps=len(A['test']) // 8, verbose=1)

    test_predicted_labels, test_actual_labels = evaluation.predict_labels(
        test_predictions, test_y, meta['idx2label'])
    test_precision, test_recall, test_f1 = evaluation.compute_scores(
        test_predicted_labels, test_actual_labels)

    print("=== Test Results ===")
    print("Precision: {:.2f}%".format(test_precision * 100))
    print("Recall: {:.2f}%".format(test_recall * 100))
    print("F1: {:.2f}".format(test_f1 * 100))
