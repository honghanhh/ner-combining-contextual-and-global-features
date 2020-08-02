import re
# build corpus
path = '/home/hanh/Videos/multiligualNER/text_gcn/'
dataset = ['conll2003/train.txt','conll2003/test.txt','conll2003/valid.txt']
for data in dataset:
    f = open(path + data,'r')
    lines = f.readlines()
    docs = []
    for line in lines:
        doc_content = line.split(" ")[0].replace('\n', ' ')
        docs.append(doc_content)

corpus_str = ' '.join(docs)
f.close()
print(len(corpus_str)) #train 1418911 #all 284868

f = open(path + 'conll2003_wv/conll_corpus.txt', 'w')
f.write(corpus_str)
f.close()
