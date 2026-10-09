import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from transformers.utils import logging

from config import *
from dataset import get_dataset

logging.disable_progress_bar()
@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    save_steps: int = SAVE_STEPS
    save_path: Path = BEST_MODEL_PATH
    judge_metric: Literal['loss', 'precision', 'recall', 'f1'] = 'f1'
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    checkpoint_path: Path = CHECKPOINT_PATH
    cast_enable: bool = True

class Trainer:
    def __init__(self, model, train_dataset, collate_fn, device, trainconfig, val_dataset=None):
        self.trainconfig = trainconfig
        self.model = model.to(device)
        self.train_dataloader = self._get_dataloader(train_dataset, collate_fn)
        self.collate_fn = collate_fn
        self.val_dataloader = self._get_dataloader(val_dataset, collate_fn) if val_dataset is not None else None
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.trainconfig.lr)
        self.step=1
        self.writer = SummaryWriter(LOG_ROOT / 'train' / time.strftime('%Y-%m-%d_%H-%M-%S'))
        self.best_score = -float('inf')
        self.score = None
        self.early_stopping_counter = 0
        self.scalar = torch.amp.GradScaler(device=self.device.type, enabled=self.trainconfig.cast_enable)

    def train(self):
        self._load_checkpoint()
        for epoch in tqdm(range(self.trainconfig.epochs), desc=f'Totel {self.trainconfig.epochs} Epochs', leave=True, position=0, ncols=100):
            for batch in tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.trainconfig.epochs}', leave=False, position=1, ncols=100):
                self.model.train()
                train_loss = self._train_step(batch)

                if self.step % self.trainconfig.save_steps == 0:
                    tqdm.write(f'Epoch {epoch+1}, Step {self.step}, Loss: {train_loss}')
                    self.writer.add_scalar('train/loss', train_loss, self.step)
                    
                    if self.val_dataloader is None:
                        self.score = -train_loss
                    else:
                        self.score = self.evaluate(self.val_dataloader)[self.trainconfig.judge_metric]

                    if self.score > self.best_score:
                        self.best_score = self.score
                        self.early_stopping_counter = 0
                        tqdm.write(f'New best score: {self.best_score}, saving model...')
                        self.model.save_pretrained(self.trainconfig.save_path)
                        self._save_checkpoint()
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.trainconfig.early_stopping_patience:
                            self.early_stopping_counter = 0
                            tqdm.write(f'Early stopping triggered. No improvement in {self.trainconfig.early_stopping_patience} evaluations.')
                            break
                self.step += 1
        self.writer.close()

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False, position=1, ncols=100):
                inputs = {
                    k: v.to(self.device) for k,v in batch.items()
                }

                outputs = self.model(**inputs)

                loss += outputs.loss.item()
                all_preds.extend(torch.argmax(outputs.logits, dim=-1))
                all_labels.extend(inputs['labels'])
            
            loss /= len(dataloader)
            metrics = self._compute_metrics(all_preds, all_labels)
        return {'loss': loss, **metrics}

    def _train_step(self, batch):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.trainconfig.cast_enable):
            inputs = {
                k:v.to(self.device) for k, v in batch.items()
            }
            loss = self.model(**inputs).loss
        self.optim.zero_grad()
        self.scalar.scale(loss).backward()
        self.scalar.step(self.optim)
        self.scalar.update()
        return loss.item()

    def _compute_metrics(self, preds, labels):
        preds = torch.cat(preds).detach().cpu()
        labels = torch.cat(labels).detach().cpu()

        valid_indices = labels != -100
        preds = preds[valid_indices]
        labels = labels[valid_indices]

        return {
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0)
        }

    def _get_dataloader(self, dataset, collate_fn):
        dataloader = DataLoader(dataset, self.trainconfig.batch_size, collate_fn=collate_fn)
        return dataloader

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'step': self.step,
            'best_score': self.best_score,
            'early_stopping_counter': self.early_stopping_counter
        }
        torch.save(checkpoint, self.trainconfig.checkpoint_path / 'checkpoint.pt')

    def _load_checkpoint(self):
        checkpoint_path = self.trainconfig.checkpoint_path / 'checkpoint.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.best_score = checkpoint['best_score']
            self.early_stopping_counter = checkpoint['early_stopping_counter']
            tqdm.write(f'Loaded checkpoint from {checkpoint_path}')
        else:
            tqdm.write(f'No checkpoint found at {checkpoint_path}, starting from scratch.')

if __name__ == '__main__':
    with open(PROCESSED_DATA_PATH / 'id2label.json', encoding='utf-8') as f:
        id2label = {int(label_id): label for label_id, label in json.load(f).items()}
    label2id = {label: label_id for label_id, label in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_PATH if pretrained_exists() else PRETRAINED_MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_PATH if pretrained_exists() else PRETRAINED_MODEL_NAME
    )
    collate_fn = DataCollatorForTokenClassification(tokenizer)
    train_dataset = get_dataset('train')
    val_dataset = get_dataset('test')
    trainconfig = TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        save_steps=SAVE_STEPS,
        save_path=BEST_MODEL_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        judge_metric='f1',
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        collate_fn=collate_fn,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        trainconfig=trainconfig,
        val_dataset=val_dataset
    )
    trainer.train()