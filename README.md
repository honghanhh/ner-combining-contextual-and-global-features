# [Internship: Multi-lingual NER taggings]

__Superviors__: 
- Prof. Nicolas __SIDERE__
- Prof. Antoine __DOUCET__
- Prof. Jose __MONERO__. 

__Student__: __TRAN__ Thi Hong Hanh.

### __Task 1__:

- [x] Implement the code using XLNet for NER training and prediction on the CONLL2003 data.
  
- [x] Reimplement on multiple datasets of NER including languages of the Embeddia project (Slovenian, Finnish, Estonian, etc.)
  - This link https://filesender.renater.fr/?s=download&token=fee352ab-62d6-945d-7a72-88d18c0dfd54 

- [x] Recommended readings:
  - [x] [__XLNet__](https://arxiv.org/pdf/1906.08237.pdf}): Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems (pp. 5754-5764).
  - [x] [__BERT__](https://arxiv.org/pdf/1810.04805.pdf): Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
  - [x] [__BERT+NER__](https://www.aclweb.org/anthology/W19-3712.pdf): Arkhipov, M., Trofimova, M., Kuratov, Y., & Sorokin, A. (2019, August). Tuning multilingual transformers for language-specific named entity recognition. In Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing (pp. 89-93).

### __Task 2__:
- Investigate multilingual word embeddings that can support our interested NER languages:
  - [x] [__FastText__](https://fasttext.cc/): 
    - Trained on __Common Crawl__ and __Wikipedia__ with 300 dimension and n_gram of length 5.
    - Support 157 languages, including English, Slovenian, Finnish, Estonian.
    - Flair supports to use FastText as word embeddings, also test in [__flair_ner__](https://github.com/honghanhh/multiligualNER/embeddings/flair_embeddings.ipynb).
  - [x] [__Flair__](https://github.com/flairNLP/flair): 
    - Support 17 languages, including English, Slovanian(419,744,423 tokens), and Finish (427,194,262 tokens).
    - List of 14 word embeddings that Flair support: [__List__](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md).
    - Can stack, mix, match word embeddings (i.e Flair, ELMo, BERT and classical one), test word embeddings in [__flair_ner__](https://github.com/honghanhh/multiligualNER/embeddings/flair_embeddings.ipynb).
  - [x] [__Stanza__](https://github.com/stanfordnlp/stanza) :
    - Support about 70 languages, including English, Slovenian, Finnish, and Estonian.

  
### __Task 3__ (20/04/2020 - 26/04/2020):
- [x] Implement XLNet to reach SOTA benchmark (at least 91% F1-score).
  - Log file in this [link](https://github.com/honghanhh/multiligualNER/bert-ner/logs/XLNet_20-02-20.out).
  
  ```
  =========eval at epoch=4=========
  processed 55044 tokens with 5942 phrases; found: 5950 phrases; correct: 5878.
  accuracy:  99.27%; (non-O)
  accuracy:  99.85%; precision:  98.79%; recall:  98.92%; FB1:  98.86
                LOC: precision:  98.55%; recall:  99.73%; FB1:  99.13  1859
              MISC: precision:  97.62%; recall:  97.72%; FB1:  97.67  923
                ORG: precision:  99.17%; recall:  97.69%; FB1:  98.42  1321
                PER: precision:  99.35%; recall:  99.62%; FB1:  99.48  1847
  num_proposed:8611
  num_correct:8540
  num_gold:8603
  precision=0.9918
  recall=0.9927
  f1=0.9922
  weights were saved to checkpoints/finetune/4.pt
  =========eval at epoch=4=========
  processed 50350 tokens with 5648 phrases; found: 5757 phrases; correct: 5204.
  accuracy:  93.80%; (non-O)
  accuracy:  98.38%; precision:  90.39%; recall:  92.14%; FB1:  91.26
                LOC: precision:  91.34%; recall:  94.84%; FB1:  93.06  1732
              MISC: precision:  81.04%; recall:  79.77%; FB1:  80.40  691
                ORG: precision:  87.67%; recall:  90.79%; FB1:  89.20  1720
                PER: precision:  96.28%; recall:  96.10%; FB1:  96.19  1614
  num_proposed:8324
  num_correct:7609
  num_gold:8112
  precision=0.9141
  recall=0.9380
  f1=0.9259
  ```
- [x] Upgrade memory in Colab, test in this [link](https://towardsdatascience.com/upgrade-your-memory-on-google-colab-for-free-1b8b18e8791d).
  ```
  a = []
  while(1):
    a.append(‘1’)
  ```
- [] Calculate predictions using [__Flair__](https://github.com/flairNLP/flair) and [__Stanza__](https://github.com/stanfordnlp/stanza).
- [] Evaluate the performance using __get_score.py__ in this [github](https://github.com/Adaxry/GCDT/tree/master/data/conll03).
- [] Learn about graph embeddings:
  - [] [Topological Feature Extractors for Named Entity Recognition using graph convolutional networks](https://www.aclweb.org/anthology/W17-7607.pdf).
  - [] [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf).
  - [] [Deeper insights into graph convolutional networks for semi-supervised learning](https://arxiv.org/pdf/1801.07606.pdf). 
  - [] [Graph convolution over pruned dependency trees improves relation extraction](https://www.aclweb.org/anthology/D18-1244.pdf). 

