import torch
import torch.nn as nn
#from pytorch_transformers import BertModel
# from pytorch_transformers import XLNetModel
#from pytorch_pretrained_bert import OpenAIGPTModel
from transformers import XLNetModel

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        #self.bert = BertModel.from_pretrained('bert-base-cased')
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        #self.bert = OpenAIGPTModel.from_pretrained('openai-gpt')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
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
            # print("->xlnet.train()")
            self.xlnet.train()
            # output1 = self.xlnet(x)
            # print("model",len(output1), output1)
            # encoded_layers = output1
            encoded_layers = self.xlnet(x)
            # print(encoded_layers.shape,len(_))
            enc = encoded_layers[0]#[-1]
        else:
            self.xlnet.eval()
            with torch.no_grad():
                encoded_layers, _ = self.xlnet(x)
                enc = encoded_layers[0]#[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
