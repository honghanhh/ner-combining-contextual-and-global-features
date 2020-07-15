import sys
import torch
import argparse
import os
from os.path import dirname, join
from data_load import NerDataset, pad, VOCAB, tag2idx, idx2tag
from model import Net
from conlleval import evaluate_conll_file

class NERTaggingPrediction:
    def __init__(self, checkpoint_path = ''):
        root_checkpoints = join(dirname(dirname(__file__)), 'finetuning')
        if checkpoint_path == '':
            self.checkpoint_path = join(root_checkpoints, '4.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning).cuda()
        self.model = nn.DataParallel(self.model)
        self.model.load_weights(self.checkpoint_path)


    def predict(self, text):
        text = NerDataset(text)
        predicted = self.model.predict(text)
        return predicted


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    hp = parser.parse_args()

    text = 'West Indian all-rounder Phil Simmons took on Friday.'
    predictor = NERTaggingPrediction()
    print(predictor.predict(text))
