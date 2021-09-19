import sys
import torch
import argparse
import os
from os.path import dirname, join
from data_load import NerDataset, pad, VOCAB, tag2idx, idx2tag
from model import Net
from conlleval import evaluate_conll_file
import torch.nn as nn
import numpy as np
from torch.utils import data
import pickle as pkl

checkpoint_path='finetuning/4.pt'
logdir = 'checkpoints/hanh'

model = torch.load(checkpoint_path)
model.eval()


newmodel = torch.nn.DataParallel(*(list(model.module.children())[:-1]))

train_dataset = NerDataset('conll2003/train.txt')
eval_dataset = NerDataset('conll2003/valid.txt')
test_dataset = NerDataset('conll2003/test.txt')

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=16,
                             num_workers=4,
                             collate_fn=pad)
eval_iter = data.DataLoader(dataset=eval_dataset,
                            batch_size=16,
                            num_workers=4,
                            collate_fn=pad)

test_iter = data.DataLoader(dataset=test_dataset,
                            batch_size=16,
                            num_workers=4,
                            collate_fn=pad)

gcn_train = pkl.load(open('../conll_gcn/pkl/train_predictions.pkl', 'rb'))
gcn_val = pkl.load(open('../conll_gcn/pkl/val_predictions.pkl', 'rb'))
gcn_test = pkl.load(open('../conll_gcn/pkl/test_predictions.pkl', 'rb'))

class EnsembleModel(nn.Module):
    def __init__(self, xln_model, gcn_pretrained, vocab_size, device = 'cuda'):
        super().__init__()
        self.xln_model = xln_model
        self.gcn_pretrained = gcn_pretrained

        self.fc = nn.Linear(768 + 256, vocab_size)

        self.device = device


    def forward(self, x, gcn_tensor):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        '''
        x = x.to(self.device)

        self.xln_model.eval()
        with torch.no_grad():
            encoded_layers = self.xln_model(x)
            enc = encoded_layers[0]
        gcn_tensor = torch.from_numpy(gcn_tensor).float()
        gcn_tensor = gcn_tensor.to(self.device)
        ensemble = torch.cat((enc, gcn_tensor), dim=2)
        logits = self.fc(ensemble)
        
        return logits


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ensemble_model = EnsembleModel(newmodel, gcn_train, vocab_size = len(VOCAB), device=device).to(device)

optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def eval(model, iterator, gcn, f):
    ensemble_model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            max_len = x.shape[1]
            batch_size = x.shape[0]
            idx = i * batch_size

            if idx+batch_size > gcn.shape[0]:
                break
            gcn_tensor = gcn[idx:idx+batch_size]
            padded = np.zeros((batch_size, max_len, 256))

            for ix in range(batch_size):
                is_heads[ix][0] = is_heads[ix][-1] = 0
                num_word = sum(is_heads[ix])
                indexs = [head == 1 for head in is_heads[ix]] + [False] * (max_len - len(is_heads[ix]))
                padded[ix][indexs]=gcn_tensor[ix][:num_word]

#             optimizer.zero_grad()

            logits = ensemble_model(x, padded)
            y_hat = logits.argmax(-1)
#             logits = logits.view(-1, logits.shape[-1])
#             y = y.to(device).view(-1)

#             _, _,y_hat = ensemble_model(x, padded)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist()) 

                 
    # gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            is_heads[0] = is_heads[-1] = 1
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
#             print(is_heads)
#             print(preds)
#             print(words.split())
#             print(tags.split())
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    with open("temp") as fout:
        evaluate_conll_file(fout)

    # calc metric
    y_true = np.array([tag2idx[line.split()[1]] for line in open(
        "temp", 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open(
        "temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred,
                                  y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.4f" % precision)
    print("recall=%.4f" % recall)
    print("f1=%.4f" % f1)
    return precision, recall, f1

for ep in range(4):
    ensemble_model.train()
    for i, batch in enumerate(train_iter):
        words, x, is_heads, _, y, seqlens = batch
        
        max_len = x.shape[1]
        batch_size = x.shape[0]
        idx = i * batch_size
        
        if idx+batch_size > gcn_train.shape[0]:
            break
        gcn_tensor = gcn_train[idx:idx+batch_size]
        padded = np.zeros((batch_size, max_len, 256))

        for ix in range(batch_size):
            is_heads[ix][0] = is_heads[ix][-1] = 0
            num_word = sum(is_heads[ix])
            indexs = [head == 1 for head in is_heads[ix]] + [False] * (max_len - len(is_heads[ix]))
            padded[ix][indexs]=gcn_tensor[ix][:num_word]
        
        optimizer.zero_grad()

        logits = ensemble_model(x, padded)

        logits = logits.view(-1, logits.shape[-1])
        y = y.to(device).view(-1)
        
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
    print(f"=========eval at epoch={ep}=========")
    if not os.path.exists(logdir): os.makedirs(logdir)
    fname = os.path.join(logdir, str(epoch))
    precision, recall, f1 = eval(ensemble_model, eval_iter, gcn_val, fname)

    torch.save(model.state_dict(), f"{fname}.pt")
    print(f"weights were saved to {fname}.pt")
    print(f"=========eval at epoch={ep}=========")
    if not os.path.exists(logdir): os.makedirs(logdir)
    fname = os.path.join(logdir, str(epoch)+'_test_')
    precision, recall, f1 = eval(ensemble_model, test_iter, gcn_test, fname)

