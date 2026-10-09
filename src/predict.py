import torch

from config import *


class Predictor:
    def __init__(self, model, tokenizer, id2label):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.id2label = id2label
        

    def predict(self, texts):
        texts = self._tokenize(texts)
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',
            is_split_into_words=True,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = inputs.pop('special_tokens_mask')
        attention_mask = inputs['attention_mask']
        inputs = inputs.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu()
            valid_mask = special_tokens_mask.eq(0) & attention_mask.eq(1)
            predlabels = [
                [self.id2label[int(idx)] for idx, valid in zip(seq, mask) if valid]
                for seq, mask in zip(predictions, valid_mask)
            ]
        return predlabels

    def _tokenize(self, texts):
        inputs = [
            [char for char in text if not char.isspace()]
            for text in texts
        ]
        return inputs