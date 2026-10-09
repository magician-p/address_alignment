from typing import Literal

from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PTDataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from config import *


class HFDataset(PTDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) :
        return self.dataset[index]

def get_dataloader(data_path, data_type, batch_size, tokenizer):
    dataset_dict = load_from_disk(str(data_path))
    dataset = dataset_dict[data_type] if isinstance(dataset_dict, DatasetDict) else dataset_dict
    
    collate_fn = DataCollatorForTokenClassification(tokenizer)
    dataset = HFDataset(dataset)
    dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn)
    return dataloader

def get_dataset(data_type:Literal['train', 'test']):
    dataset_dict = load_from_disk(str(PROCESSED_DATA_PATH / data_type))
    dataset = dataset_dict[data_type] if isinstance(dataset_dict, DatasetDict) else dataset_dict
    dataset = HFDataset(dataset)
    return dataset


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        str(PRETRAINED_MODEL_PATH)
        if pretrained_exists() else str(PRETRAINED_MODEL_NAME)
    )
    dataloader = get_dataloader(PROCESSED_DATA_PATH, 'train', BATCH_SIZE, tokenizer)
    
    batch = next(iter(dataloader))
    for k, v in batch.items():
        print(f'{k}: {v}')