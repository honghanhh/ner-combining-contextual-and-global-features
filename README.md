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
  - [x] __FastText__: 
    - Trained on __Common Crawl__ and __Wikipedia__ with 300 dimension and n_gram of length 5.
    - Support 157 languages, including English, Slovenian, Finnish, Estonian.
    - Flair supports to use FastText as word embeddings, also test in [__flair_ner__](https://github.com/honghanhh/multiligualNER/embeddings/flair/).
  - [x] __Flair__: 
    - Support 17 languages, including English, Slovanian(419,744,423 tokens), and Finish (427,194,262 tokens).
    - List of 14 word embeddings that Flair support: [__List__](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md).
    - Can stack, mix, match word embeddings (i.e Flair, ELMo, BERT and classical one), test word embeddings in [__flair_ner__](https://github.com/honghanhh/multiligualNER/embeddings/flair/).
  - [x] __Stanza__:
