import json

from datasets import ClassLabel, Sequence, load_dataset
from transformers import AutoTokenizer

from config import *


def preprocess():
    dataset = load_dataset('json', data_files={'train': str(RAW_DATA_PATH)})
    dataset = dataset.filter(lambda x: x['text'] is not None)
    
    all_labels = sorted({
        label 
        for item in dataset.values()
        for labels in item['labels']
        for label in labels
    })
    
    with open(PROCESSED_DATA_PATH / 'id2label.json', 'w', encoding='utf-8') as f:
        json.dump({
            idx: label
            for idx, label in enumerate(all_labels)
        }, f, ensure_ascii=False, indent=4)

    labels_feature = Sequence(ClassLabel(names=all_labels))
    dataset = dataset.cast_column('labels', labels_feature)
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(PRETRAINED_MODEL_PATH) if pretrained_exists() else str(PRETRAINED_MODEL_NAME)
    )

    def decode(batch):
        inputs = tokenizer(batch['text'], is_split_into_words=True, truncation=True)
        aligned_labels = []
        for batch_index, labels in enumerate(batch['labels']):
            word_ids = inputs.word_ids(batch_index=batch_index)
            label = []
            for word_id in word_ids:
                if word_id is None:
                    label.append(-100)
                else:
                    label.append(labels[word_id])
            aligned_labels.append(label)
        inputs['labels'] = aligned_labels
        return inputs
    
    dataset = dataset.map(decode, batched=True, remove_columns=['text'])
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset.save_to_disk(str(PROCESSED_DATA_PATH))


if __name__ == '__main__': 
    preprocess()