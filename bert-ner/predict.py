import sys
import torch

class NERTaggingPrediction:
    def __init__(self, checkpoint_path = ''):
        root_checkpoints = join(dirname(dirname(__file__)), 'finetune')
        if checkpoint_path == '':
            self.checkpoint_path = join(root_checkpoints, '')

    def predict():
        return True

if __name__ == '__main__':
    text = ''
    predictor = NERTaggingPrediction()
    print(predictor.predict(text))
