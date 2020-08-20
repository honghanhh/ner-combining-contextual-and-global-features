import re
# build corpus
path = '/home/hanh/Videos/multiligualNER/text_gcn/'
dataset = ['conll2003/train.txt','conll2003/valid.txt','conll2003/test.txt']
corpus = []
for data in dataset:
    print(data)
    f = open(path + data,'r')
    lines = f.readlines()
    docs = []
    for line in lines:
        doc_content = line.split(" ")[0].replace('\n', ' ')
        docs.append(doc_content)
    corpus.append(' '.join(docs))

corpus_str = ' '.join(corpus)
f.close()
print(len(corpus_str)) #train 1418911 #valid 284868 #test 256457 #all 1960238

f = open(path + 'conll2003_wv/conll_corpus.txt', 'w')
f.write(corpus_str)
f.close()