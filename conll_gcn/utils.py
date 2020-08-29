from keras.utils import to_categorical

import gzip
import numpy as np
import scipy.sparse as sp

import spacy
from spacy.tokens import Doc

nlp = spacy.load('en_core_web_sm')

np.random.seed(123)


class GraphPreprocessor():

    def __init__(self, word2idx, case_sensitive=True):
        self.splits = {}
        self.relations = []
        self.max_num_nodes = 0
        self.word2idx = word2idx
        self.label2idx = {}
        self.case_sensitive = case_sensitive

    def add_split(self, filepath, name):
        sentences, sentences_labels = read_conll(filepath)
        print("Generating dependency graphs for {} split...".format(name))
        sentences_dependency_triples = self._sentences_dependency_triples(
            sentences)

        self.splits[name] = {'sentences': sentences, 'sentences_dependency_triples':
                             sentences_dependency_triples, 'sentences_labels': sentences_labels}

        self._update_relations()
        self._update_max_num_nodes()
        self._update_label2idx()

    def get_split(self, name):
        if name in self.splits:
            return self.splits[name]
        return None

    def _sentences_dependency_triples(self, sentences):
        sentences_dependency_triples = []
        for i, sentence in enumerate(sentences):
            print("", end='\r')
            print("Generating dependency graph {}/{}...".format(i +
                                                                1, len(sentences)), end='')
            sentences_dependency_triples.append(
                self._dependency_triples(sentence))
        print("Done")
        return sentences_dependency_triples

    def _dependency_triples(self, sentence):
        doc = Doc(nlp.vocab, words=sentence)
        result = nlp.parser(doc)
        triples = []
        for token in result:
            triples.append((token.i, token.dep_, token.head.i))
        return triples

    def _sentence_dicts(self, split_name):
        sentence_dicts = []
        for i in range(len(self.splits[split_name]['sentences'])):
            sentence_dict = {
                'sentence': self.splits[split_name]['sentences'][i],
                'sentence_dependency_triples': self.splits[split_name]['sentences_dependency_triples'][i],
                'sentence_labels': self.splits[split_name]['sentences_labels'][i]
            }
            sentence_dicts.append(sentence_dict)
        return sentence_dicts

    def _sentence_adjacency_matrices(self, sentence_dict, symmetric_normalization):
        adj_matrices = []
        for relation in self.relations:
            adj_matrix = sp.lil_matrix(
                (self.max_num_nodes, self.max_num_nodes), dtype='int8')
            for triple in sentence_dict['sentence_dependency_triples']:
                if triple[1] == relation:
                    adj_matrix[triple[0], triple[2]] = 1
            adj_matrix = adj_matrix.tocsr()
            if symmetric_normalization:
                adj_matrix = self._symmetric_normalization(adj_matrix)
            adj_matrices.append(adj_matrix)
        return adj_matrices

    def _update_relations(self):
        all_relations = []
        for split_name in self.splits.keys():
            for sentence_triples in self.splits[split_name]['sentences_dependency_triples']:
                for triple in sentence_triples:
                    all_relations.append(triple[1])
        self.relations = list(set(all_relations))

    def _update_max_num_nodes(self):
        all_lengths = []
        for split_name in self.splits.keys():
            for sentence in self.splits[split_name]['sentences']:
                all_lengths.append(len(sentence))
        self.max_num_nodes = max(all_lengths)

    def _update_label2idx(self):
        all_labels = []
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_labels']:
                for label in sentence_labels:
                    all_labels.append(label)
        unique_labels = list(set(all_labels))
        self.label2idx = {v: i for i, v in enumerate(unique_labels)}

    def _lookup_sentence(self, sentence):
        tokens = [0 for x in range(self.max_num_nodes)]
        for i, word in enumerate(sentence):
            if not self.case_sensitive:
                word = word.lower()
            if word in self.word2idx:
                tokens[i] = self.word2idx[word]
        tokens = sp.csr_matrix(np.array(tokens))
        return tokens

    def _lookup_sentence_labels(self, sentence_labels):
        label_idx = [self.label2idx['O'] for x in range(self.max_num_nodes)]
        for i, label in enumerate(sentence_labels):
            label_idx[i] = self.label2idx[label]
        label_idx = sp.csr_matrix(np.array(label_idx))
        return label_idx

    def _symmetric_normalization(self, A):
        d = np.array(A.sum(1)).flatten()
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        return D_inv.dot(A)

    def input_data(self):
        input_data = {k: [] for k in self.splits.keys()}
        for split_name in self.splits.keys():
            for sentence in self.splits[split_name]['sentences']:
                tokens = self._lookup_sentence(sentence)
                input_data[split_name].append(tokens)
            input_data[split_name] = np.array(input_data[split_name])
        return input_data

    def output_data(self):
        output_data = {k: [] for k in self.splits.keys()}
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_labels']:
                label_idx = self._lookup_sentence_labels(sentence_labels)
                output_data[split_name].append(label_idx)
            output_data[split_name] = np.array(output_data[split_name])
        return output_data

    def adjacency_matrices(self, symmetric_normalization=True):
        A = {k: [] for k in self.splits.keys()}
        node_ids = []
        for split_name in self.splits.keys():
            print("Generating adjacency matrix for {} split...".format(split_name))
            for i, sentence_dict in enumerate(self._sentence_dicts(split_name)):
                print("", end='\r')
                print("Generating adjacency matrix {}/{}...".format(i +
                                                                    1, len(self.splits[split_name]['sentences'])), end='')

                adjacency_matrix = self._sentence_adjacency_matrices(
                    sentence_dict, symmetric_normalization=symmetric_normalization)
                A[split_name].append(adjacency_matrix)
            print("Done")
        return A


def read_conll(filepath):
    sentences = []
    sentences_labels = []
    with open(filepath, "r") as f:
        sentence = []
        sentence_labels = []
        for i, line in enumerate(f):
            # Only allow ASCII characters
            line = ''.join(i for i in line if ord(i) < 128)
            line_split = line.split()
            if len(line_split) == 0:
                sentences.append(sentence)
                sentence = []
                sentences_labels.append(sentence_labels)
                sentence_labels = []
            else:
                sentence.append(line_split[0])
                sentence_labels.append(line_split[3])
    if len(sentence) > 0:
        sentences.append(sentence)
        sentences_labels.append(sentence_labels)
    return sentences, sentences_labels


def word2idx_from_embeddings(embeddings_str, max_num_words=None):
    with gzip.open(embeddings_str, 'rb+') as f:
        word2idx = {}
        tokens = []
        for line in f:
            if max_num_words is not None and len(tokens) >= max_num_words:
                break
            line_split = line.split()
            tokens.append(line_split[0].decode('utf-8'))
        word2idx = {v: k for k, v in enumerate(tokens)}
        return word2idx


def matrix_dimension(embeddings_str):
    with gzip.open(embeddings_str, 'rb+') as f:
        line = next(f)
        return len(line.split()) - 1


def matrix_from_embeddings(embeddings_str, word2idx):
    embedding_matrix = np.zeros(
        (len(word2idx), matrix_dimension(embeddings_str)))
    with gzip.open(embeddings_str, 'rb+') as f:
        for line in f:
            line_split = line.split()
            token = line_split[0].decode('utf-8')
            token = token.lower()
            idx = word2idx.get(token, 0)
            if idx > 0:
                embedding_vector = np.asarray(line_split[1:], dtype='float32')
                embedding_matrix[idx] = embedding_vector
        return embedding_matrix


def load_data(A, X, Y, split_name):
    split_x = [[] for x in A[split_name][0]]
    split_y = []

    split_x[0] = [x.toarray()[0] for x in X[split_name]]
    for i in range(len(A[split_name][0]) - 1):
        for j in range(len(A[split_name])):
            split_x[i + 1].append(A[split_name][j][i].toarray())

    split_y = [to_categorical(y.toarray()[0], num_classes=8)
               for y in Y[split_name]]

    return split_x, split_y


def load_output(A, X, Y, split_name):
    split_y = [to_categorical(y.toarray()[0], num_classes=8)
               for y in Y[split_name]]

    return split_y


def batch_generator(A, X, Y, split_name, batch_size=16):
    num_sentences = len(X[split_name])
    batch_counter = 0

    split_x = [[] for x in A[split_name][0]]
    split_y = []

    while True:
        batch_start = batch_counter * batch_size
        batch_end = (batch_counter + 1) * batch_size
        split_x[0] = np.array([x.toarray()[0]
                               for x in X[split_name][batch_start:batch_end]])
        for i in range(len(A[split_name][0]) - 1):
            for j in range(len(A[split_name][batch_start:batch_end])):
                split_x[i + 1].append(A[split_name]
                                      [batch_start:batch_end][j][i].toarray())
            split_x[i + 1] = np.array(split_x[i + 1])
        split_y = np.array([to_categorical(y.toarray()[0], num_classes=8)
                            for y in Y[split_name][batch_start:batch_end]])
        batch_counter = (batch_counter + 1) % (num_sentences // batch_size)
        yield split_x, split_y
        split_x = [[] for x in A[split_name][0]]
        split_y = []
