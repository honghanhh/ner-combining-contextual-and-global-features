import torch
import torch.nn as nn
from transformers import XLNetModel

'''
BertModel: 'bert-base-cased'
OpenAIGPTModel: 'openai-gpt'
XLNetModel:'xlnet-base-cased'
'''


class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')

        self.top_rnns = top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
                               input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            self.xlnet.train()
            encoded_layers = self.xlnet(x)
            enc = encoded_layers[0]
        else:
            self.xlnet.eval()
            with torch.no_grad():
                encoded_layers = self.xlnet(x)
                enc = encoded_layers[0]

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
