# Spectral Graph Convolutional Networks for Sequence Labelling in Keras

This project implements Spectral Graph Convolutional Networks (Kipf and Welling, 2016) for sequence labelling tasks. The system achieves 82.69 F1 score on the CoNLL-2003 named entity recognition dataset. This implementation is based on Thomas Kipf's GCN implementation for relational graphs, but is designed for text sequence labelling and works with Tensorflow and Keras 2. The evaluation code is based on: https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf. I've written a blog post about the project here: https://jordanhart.co.uk/2018/09/21/spectral-graph-convolutional-networks-for-sequence-labelling-in-keras/.

## Requirements

* Tensorflow 1.7.0
* SciPy 1.1.0
* Spacy 2.0.11
* Keras 2.2.0
* NumPy 1.13.3

## Usage - Sequence labelling

To run the system on the CoNLL-2003 NER dataset (https://www.clips.uantwerpen.be/conll2003/ner/) with Komninos' embeddings (https://www.cs.york.ac.uk/nlp/extvec/), first, create a directory in `data` with the name of the dataset, say `conll2003`. Place the CoNLL format dataset files `train.txt`, `val.txt`, and `test.txt` into the `data/conll2003` directory and place Komninos' gzipped embeddings into the `embeddings` directory.

To prepare the dataset, run

```
python prepare_dataset.py -d conll2003 -e komninos_english_embeddings
```

This will take a few minutes, and output two files into the `pkl` folder: `conll2003.pkl` and `conll2003.embedding_matrix.pkl`. Run `python prepare_dataset.py --help` for more information about how to use this script.
 
Once finished, the model can be trained and evaluated by running

```
python train.py
```

The system will train significantly faster on a graphics card, but uses a large amount of memory. To see which hyperparameters can be changed, run `python train.py --help`.

## Usage - `SpectralGraphConvolution` layer

The `SpectralGraphConvolution` layer performs spectral convolutions on graph features. For more information, the original paper can be found here: https://arxiv.org/abs/1609.02907. 

The layer can be imported by calling

```
from layers.graph import SpectralGraphConvolution
```

The layer can be included by calling

```
SpectralGraphConvolution(output_dim)
```

### Arguments

* `output_dim` - integer, the number of dimensions of the feature vectors for each node.
* `init` - initializer, the initializer used to initialize the internal weights.
* `activation` - activation, the activation function to use.
* `W_regularizer` - regularizer, the regularization function to use for the internal weights.
* `b_regularizer` - regularizer, the regularization function to use for the internal bias.
* `bias` - boolean, whether or not to include a bias term.
* `self_links` - boolean, connects each node to itself under a new relation label.
* `consecutive_links` - boolean, connects each word node to the next word node in the sentence under a new relation label.
* `backward_links` - boolean, adds a new set of relations which is the reverse of the current set of relations.
* `edge_weighting` - boolean, performs an element-wise product to assign weighting to the graph edges.

### Inputs

A 3D tensor with shape `(batch_size, num_nodes, num_features)` and a list of 3D tensors with shape `(batch_size, num_nodes, num_nodes)`.

### Outputs

A 3D tensor with shape `(batch_size, num_nodes, output_dim)`.

### Example

This excerpt is taken from `train.py`.

```
X_in = Input(shape=(num_nodes, ))
A_in = [Input(shape=(num_nodes, num_nodes)) for _ in range(num_relations)]

# Define model architecture
X_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(X_in)
H = SpectralGraphConvolution(150, activation='relu')([X_embedding] + A_in)
H = Dropout(DO)(H)
output = SpectralGraphConvolution(num_labels, activation='softmax', backward_links=False)([H] + A_in)
model = Model(inputs=[X_in] + A_in, outputs=output)
```