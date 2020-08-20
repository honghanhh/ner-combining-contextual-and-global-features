def countOccurences(word,context_window): 

    """
    This function returns the count of context word.
    """ 
    return context_window.count(word)

def co_occurance(feature_dict,corpus,window = 5):
    """
    This function returns co_occurance matrix for the given window size. Default is 5.

    """
    length = len(feature_dict)
    co_matrix = np.zeros([length,length]) # n is the count of all words

    corpus_len = len(corpus)
    for focus_word in top_features:

        for context_word in top_features[top_features.index(focus_word):]:
            # print(feature_dict[context_word])
            if focus_word == context_word:
                co_matrix[feature_dict[focus_word],feature_dict[context_word]] = 0
            else:
                start_index = 0
                count = 0
                while(focus_word in corpus[start_index:]):

                    # get the index of focus word
                    start_index = corpus.index(focus_word,start_index)
                    fi,li = max(0,start_index - window) , min(corpus_len-1,start_index + window)

                    count += countOccurences(context_word,corpus[fi:li+1])
                    # updating start index
                    start_index += 1

                # update [Aij]
                co_matrix[feature_dict[focus_word],feature_dict[context_word]] = count
                # update [Aji]
                co_matrix[feature_dict[context_word],feature_dict[focus_word]] = count
    return co_matrix