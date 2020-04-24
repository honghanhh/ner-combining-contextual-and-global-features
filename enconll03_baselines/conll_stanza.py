import stanza
import sys
stanza.download('en', processors={'tokenize': 'ewt', 'ner': 'conll03'}, package=None)

def getStanzaLabels(doc,ents):
    print("#",end='')
    pos2tok =[]
    for i, tok in enumerate(doc):
        pos2tok.extend([i]*len(tok))
    i=0
    labels = []
    impresotypes = {'ORG':'ORG','PER':'PER','LOC':'LOC','MISC':'MISC'}
    for ent in ents:
        toks = sorted(list(set(pos2tok[ent.start_char:ent.end_char])))
        while i<toks[0]:
            labels.append('O')
            i+=1
        enttype = ent.type
        labels.append(enttype if enttype != 'O' else 'O')
        i+=1
        for _ in range(len(toks)-1):
            labels.append(enttype if enttype != 'O' else 'O')
            i+=1

    while len(labels)< len(doc):
        labels.append('O')
    
    return labels


dataset = sys.argv[1]

nlp = stanza.Pipeline(lang='en', processors={'tokenize':'ewt','ner':'conll03'}, tokenize_pretokenized=True)
with open(dataset, "rt") as f_p:
    lines = []
    markedtypes = {'ORG':'I-ORG','PER':'I-PER','LOC':'I-LOC','MISC':'I-MISC','O':'O'}
    for line in f_p:
        line = line.rstrip()

        if not line:
            newlines = []
            if len(lines)>0:
              textdoc = [x[0]+" " for x in lines]
              doc = nlp("".join(textdoc))
              y_pred = getStanzaLabels(textdoc,[ent for sent in doc.sentences for ent in sent.ents])
#            print(textdoc,y_pred,"".join(textdoc))
              for i,(l,y) in enumerate(zip(lines,y_pred)):
                 newlines.append([l[0],markedtypes[y]])
            with open(dataset+".stanza.new", "a+") as f_o:
                f_o.write("\n".join([" ".join(x) for x in newlines]))
                f_o.write("\n\n" if len(lines)>0 else "\n")
            newlines = []
            lines = []
            continue

        lines.append(line.split())
