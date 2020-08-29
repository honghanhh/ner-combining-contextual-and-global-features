# [Internship: Multi-lingual NER taggings]

**Superviors**:

- Prof. Nicolas **SIDERE**
- Prof. Antoine **DOUCET**
- Prof. Jose **MORENO**.

**Student**: **TRAN** Thi Hong Hanh.

### **Task 1**:

- [x] Implement the code using XLNet for NER training and prediction on the CONLL2003 data.

- [x] Reimplement on multiple datasets of NER including languages of the Embeddia project (Slovenian, Finnish, Estonian, etc.)

  - This link https://filesender.renater.fr/?s=download&token=fee352ab-62d6-945d-7a72-88d18c0dfd54

- [x] Recommended readings:
  - [x] [**XLNet**](https://arxiv.org/pdf/1906.08237.pdf}): Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems (pp. 5754-5764).
  - [x] [**BERT**](https://arxiv.org/pdf/1810.04805.pdf): Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
  - [x] [**BERT+NER**](https://www.aclweb.org/anthology/W19-3712.pdf): Arkhipov, M., Trofimova, M., Kuratov, Y., & Sorokin, A. (2019, August). Tuning multilingual transformers for language-specific named entity recognition. In Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing (pp. 89-93).

### **Task 2**:

- Investigate multilingual word embeddings that can support our interested NER languages:
  - [x] [**FastText**](https://fasttext.cc/):
    - Trained on **Common Crawl** and **Wikipedia** with 300 dimension and n_gram of length 5.
    - Support 157 languages, including English, Slovenian, Finnish, Estonian.
    - Flair supports to use FastText as word embeddings, also test in [**flair_ner**](https://github.com/honghanhh/multiligualNER/embeddings/flair_embeddings.ipynb).
  - [x] [**Flair**](https://github.com/flairNLP/flair):
    - Support 17 languages, including English, Slovanian(419,744,423 tokens), and Finish (427,194,262 tokens).
    - List of 14 word embeddings that Flair support: [**List**](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md).
    - Can stack, mix, match word embeddings (i.e Flair, ELMo, BERT and classical one), test word embeddings in [**flair_ner**](https://github.com/honghanhh/multiligualNER/embeddings/flair_embeddings.ipynb).
  - [x] [**Stanza**](https://github.com/stanfordnlp/stanza) :
    - Support about 70 languages, including English, Slovenian, Finnish, and Estonian.

### **Task 3** (20/04/2020 - 26/04/2020):

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

  - Saved logs in this [link](https://github.com/honghanhh/multiligualNER/bert-ner/finetune/)
  - Saved checkpoints in this [link](https://github.com/honghanhh/multiligualNER/bert-ner/checkpoints/)

- [x] Upgrade memory in Colab, test in this [link](https://towardsdatascience.com/upgrade-your-memory-on-google-colab-for-free-1b8b18e8791d).
  ```
  a = []
  while(1):
    a.append(‘1’)
  ```
  - It works !!!!
- [x] Calculate predictions using [**Flair**](https://github.com/flairNLP/flair) and [**Stanza**](https://github.com/stanfordnlp/stanza).
  - Prediction result files of each approach are in [eng.testb.2.examples.txt.flair.new](https://github.com/honghanhh/multiligualNER/enconll03_baselines/eng.testb.2.examples.txt.flair.new) and [eng.testb.2.examples.txt.stanza.new](https://github.com/honghanhh/multiligualNER/enconll03_baselines/eng.testb.2.examples.txt.stanza.new), respectively.
  - Note of stanza model:
  ```
  stanza.download('en', processors={'tokenize': 'ewt', 'ner': 'conll03'})
  ```
- [x] Evaluate the performance using **get_score.py** in this [github](https://github.com/Adaxry/GCDT/tree/master/data/conll03).
  - Evaluating data: [eng.testb.2.examples.txt](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.testb.2.examples.txt).
  - **Stanza**:
    - Generate **510/50120** bad datas.
    - Predicting result is saved in [eng.testb.2.examples.txt.stanza.new](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.testb.2.examples.txt.stanza.new).
    - Wrong predictions are collected in [eng.res.stanza.txt](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.res.stanza.txt).
  - **Flair**:
    - Generate **20151/50120** bad datas.
    - Predicting result is saved in [eng.testb.2.examples.txt.flair.new](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.testb.2.examples.txt.flair.new).
    - Wrong predictions are collected in [eng.res.flair.txt](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.res.flair.txt).
- Recommend papers about graph embeddings:
  - [x] Graph Convolutional Networks for Named Entity Recognition - 2017.
    - Paper link: [GCN](https://www.aclweb.org/anthology/W17-7607.pdf).
    - Source code: [gcn_ner](https://github.com/contextscout/gcn_ner).
    - Keynotes: - Input vectors = Morphological embeddings + POS embeddings + word embeddings. - The bi-directional architectures: (a) LSTM and (b) GCN:
      ![Bi-directional architectures: (a) LSTM; and (b) GCN](images/bi_LSTM-bi_GCN.png) - Dependency trees play a positive role for entity recognition by using a GCN to boost the
      results of a bidirectional LSTM.
  - [x] Deeper insights into graph convolutional networks for semi-supervised learning - 2018.
    - Paper link: [GCN](https://arxiv.org/pdf/1801.07606.pdf).
    - Source code:
    - Keynotes:
      - Use GCN for **semi-supervised learning**.
      - Pros:
        - The graph convolution – Laplacian smoothing helps making the classification problem much easier
        - The multi-layer neural network is a powerful feature extractor.
        - Cons:
          - The graph convolution is a localized filter, which performs unsatisfactorily with few labeled data.
          - The neural network needs considerable amount of labeled data for validation and model selection.
      - Solutions: **co-training** vs **self-training** GCN (spatial vs spectral).
      - Graph convolution of the GCN model is actually a special form of Laplacian smoothing (mix the features of a vertex and its nearby neighbors).
  - Semi-supervised classification with graph convolutional networks - 2017.
    - Paper link: [GCN](https://arxiv.org/pdf/1609.02907.pdf).
    - Source code:
    - Keynotes:
      - Updating
  - [x] Graph convolution over pruned dependency trees improves relation extraction.
    - Paper link: [GCN](https://www.aclweb.org/anthology/D18-1244.pdf).
    - Source code:
    - Keynotes:
      - Extract relation using GCN to efficiently pool information over arbitrary dependency structures.
      - Use new pathcentric pruning technique to help dependency-based models maximally remove irrelevant information without damaging crucial content to improve their robustness.
      - The architecture:
        ![Relation extraction with a graph convolutional network.](images/architecture_GCN.png)

### **Task 4**:

- [x] Evaluate XLNetNER using new get_score.py

  - Due to CUDA memory error, I have not saved the checkpoint with tunned parameters yet.

  ```
  Traceback (most recent call last):
  File "train.py", line 164, in <module>
  train(model, train_iter, optimizer, criterion)
    ...
  File "/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py", line 936, in dropout
  else _VF.dropout(input, p, training))
  RuntimeError: CUDA out of memory. Tried to allocate 68.00 MiB (GPU 0; 11.17 GiB total capacity; 10.01 GiB already allocated; 58.81 MiB free; 10.74 GiB reserved in total by PyTorch)
  ```

  - Current XLNetNER:
    - Command:
    ```
    python3.6 train.py --logdir $SLURM_JOB_ID/finetuning --finetuning --batch_size 16 --lr 5e-5 --n_epochs 4
    ```
    - Result logs and checkpoints are saved in [Google Drive](https://drive.google.com/drive/folders/1yWKrjyTj_5ETjG-ppF2Xr43hVhYz0_kk?usp=sharing).
    ```
    =========eval at epoch=4=========
    processed 55044 tokens with 5942 phrases; found: 5944 phrases; correct: 5823.
    accuracy:  98.80%; (non-O)
    accuracy:  99.75%; precision:  97.96%; recall:  98.00%; FB1:  97.98
                  LOC: precision:  98.85%; recall:  98.26%; FB1:  98.55  1826
                MISC: precision:  95.06%; recall:  96.10%; FB1:  95.58  932
                  ORG: precision:  97.02%; recall:  97.17%; FB1:  97.09  1343
                  PER: precision:  99.24%; recall:  99.29%; FB1:  99.27  1843
    num_proposed:8596
    num_correct:8500
    num_gold:8603
    precision=0.9888
    recall=0.9880
    f1=0.9884
    weights were saved to checkpoints/finetuning/4.pt
    =========eval at epoch=4=========
    processed 50350 tokens with 5648 phrases; found: 5696 phrases; correct: 5111.
    accuracy:  92.14%; (non-O)
    accuracy:  98.21%; precision:  89.73%; recall:  90.49%; FB1:  90.11
                  LOC: precision:  94.67%; recall:  91.55%; FB1:  93.08  1613
                MISC: precision:  76.70%; recall:  83.48%; FB1:  79.95  764
                  ORG: precision:  85.59%; recall:  87.24%; FB1:  86.40  1693
                  PER: precision:  95.26%; recall:  95.79%; FB1:  95.53  1626
    num_proposed:8199
    num_correct:7474
    num_gold:8112
    precision=0.9116
    recall=0.9214
    f1=0.9164
    ```

- [x] Double check stanza and flair baselines.

  - Data: [eng.testb.2.examples.txt](https://github.com/honghanhh/multiligualNER/enconll03_baselines/result/eng.testb.2.examples.txt).
  - [x] Flair:
    - Syntax:
    ```
    python get_score.py -predict_file ./result_baselines/eng.testb.2.examples.txt.flair.new -golden_file eng.testb.2.examples.txt -result_file ./result_baselines/flair.txt
    ```
    - Result:
    ```
    Generate 510/50120 bad datas when evaluating ./result_baselines/eng.testb.2.examples.txt.flair.new processed 45925 tokens with 5446 phrases; found: 5428 phrases; correct: 5275.
    accuracy:  99.56%; precision:  97.18%; recall:  96.86%; FB1:  97.02
                LOC: precision:  96.49%; recall:  96.14%; FB1:  96.32  1626
              MISC: precision:  98.67%; recall:  97.37%; FB1:  98.01  600
                ORG: precision:  95.78%; recall:  95.60%; FB1:  95.69  1612
                PER: precision:  98.74%; recall:  98.68%; FB1:  98.71  1590
    ```
  - [x] Stanza:

    - Syntax:

    ```
    python get_score.py -predict_file ./result_baselines/eng.testb.2.examples.txt.stanza.new -golden_file eng.testb.2.examples.txt -result_file ./result_baselines/stanza.txt
    ```

    - Result:

    ```
    Generate 577/50120 bad datas when evaluating ./result_baselines/eng.testb.2.examples.txt.stanza.new
    processed 45858 tokens with 5420 phrases; found: 5397 phrases; correct: 5230.
    accuracy:  99.50%; precision:  96.91%; recall:  96.49%; FB1:  96.70
                  LOC: precision:  96.02%; recall:  95.90%; FB1:  95.96  1632
                MISC: precision:  98.67%; recall:  97.37%; FB1:  98.01  600
                  ORG: precision:  95.33%; recall:  95.03%; FB1:  95.18  1584
                  PER: precision:  98.73%; recall:  98.24%; FB1:  98.49  1581

    ```

### **Task 5**: Integrating graph embeddings into an NER system

- [x] NER using contextual embeddings only.
  ````
    =========eval at epoch=4=========
    processed 55044 tokens with 5942 phrases; found: 5944 phrases; correct: 5823.
    accuracy:  98.80%; (non-O)
    accuracy:  99.75%; precision:  97.96%; recall:  98.00%; FB1:  97.98
                  LOC: precision:  98.85%; recall:  98.26%; FB1:  98.55  1826
                MISC: precision:  95.06%; recall:  96.10%; FB1:  95.58  932
                  ORG: precision:  97.02%; recall:  97.17%; FB1:  97.09  1343
                  PER: precision:  99.24%; recall:  99.29%; FB1:  99.27  1843
    num_proposed:8596
    num_correct:8500
    num_gold:8603
    precision=0.9888
    recall=0.9880
    f1=0.9884
    weights were saved to checkpoints/finetuning/4.pt
    =========eval at epoch=4=========
    processed 50350 tokens with 5648 phrases; found: 5696 phrases; correct: 5111.
    accuracy:  92.14%; (non-O)
    accuracy:  98.21%; precision:  89.73%; recall:  90.49%; FB1:  90.11
                  LOC: precision:  94.67%; recall:  91.55%; FB1:  93.08  1613
                MISC: precision:  76.70%; recall:  83.48%; FB1:  79.95  764
                  ORG: precision:  85.59%; recall:  87.24%; FB1:  86.40  1693
                  PER: precision:  95.26%; recall:  95.79%; FB1:  95.53  1626
    num_proposed:8199
    num_correct:7474
    num_gold:8112
    precision=0.9116
    recall=0.9214
    f1=0.9164
    ```
  ````
- [x] NER using only graph embeddings.

  ````
  === EPOCH 4 ===
  Epoch 1/1
  2306/2306 [==============================] - 2361s 1s/step - loss: 0.0112
  433/433 [==============================] - 106s 246ms/step
  === Validation Results ===
      Weighted F1-score:  0.998686087337719
      Classification report:
                  precision    recall  f1-score   support

              LOC       0.98      0.94      0.96      2178
              MISC      0.86      0.96      0.91      1137
              O         1.00      1.00      1.00    421185
              ORG       0.89      0.97      0.93      1924
              PER       0.98      0.99      0.98      3112

        accuracy                           1.00    429536
        macro avg       0.94      0.97      0.96    429536
     weighted avg       1.00      1.00      1.00    42953

      === Test Results ===
      Weighted F1-score:  0.9967200065416499
      Classification report:
                  precision    recall  f1-score   support

              LOC       0.91      0.85      0.88      2066
              MISC      0.71      0.80      0.75       804
              O         1.00      1.00      1.00    448555
              ORG       0.77      0.87      0.82      2201
              PER       0.94      0.96      0.95      2694

          accuracy                          1.00    456320
          macro avg     0.86      0.90      0.88    456320
        weighted avg    1.00      1.00      1.00    456320

      ```
  ````

- [] Combining contextual embeddings and graph embeddings: On progress
