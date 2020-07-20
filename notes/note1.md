## Integrating graph embeddings into an NER system

Named entity recognition (NER) is the task of identifying text phrases that mention persons, places or organizations. 
Correctly identifying this phrases is a hard task because:
1. They have multiple forms
2. They are context dependent

Context can strongly represented by contextual embeddings. However, global relations are misrepresented by those models. Including global features may deal with better NER predictions.


### Motivation:
- Use of global information -> There are tokens that are always part of an entity? Yes, think to *France* as an unique token or *Saint*. However it is not true to all tokens in an entity. For example the tokens “The” or “White” may be or not part of an entity. In train data of the Conll03 dataset, the token *Cup* appears 95 time and all cases correspond to entity mentions although *Cup* is not obviously an entity word. More obvious cases are *U.S.* (377 mentions), *Germany* (143 mentions), *Australia* (136 mentions), *Britain* (133 mentions), *England* (127 mentions), *France* (127 mentions), to mention a few. More examples can be found in [this notebook](https://colab.research.google.com/drive/1IEvRO8ETDFnbLImDLyHlftu9BhlND5uc?usp=sharing)
- Use of local but contextual information -> SoTA algorithms

### Experiments:
1. NER using contextual embeddings only.
2. NER using only token-based embeddings classification based on [TextGCN](https://github.com/yao8839836/text_gcn).
3. Combining contextual embeddings and TextGCN token classification.

![Architecture](images/XLNetTextGCN.png)

