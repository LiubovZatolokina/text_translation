import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class OpusDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.X = self.dataset.en
        self.y = self.dataset.fr
        self.source_len = 128
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ctext = str(self.X[index])
        ctext = ' '.join(ctext.split())

        text = str(self.y[index])
        text = ' '.join(text.split())

        source_en = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')
        source_fr = self.tokenizer.batch_encode_plus([text], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')

        source_en_ids = source_en['input_ids'].squeeze()
        source_fr_ids = source_fr['input_ids'].squeeze()

        return {
            'source_en_ids': source_en_ids.to(dtype=torch.long),
            'source_fr_ids': source_fr_ids.to(dtype=torch.long)
        }


def prepare_data_for_training():
    train_data = OpusDataset('data/train.csv')
    test_data = OpusDataset('data/test.csv')

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)
    return train_loader, test_loader


def load_data():
    dataset = load_dataset('opus_books', 'en-fr')
    length = len(dataset['train'])
    print(length)
    substr = 'translate English to French: '
    dataset = dataset.shuffle()
    train_dataset = pd.DataFrame(columns=['en', 'fr'])
    test_dataset = pd.DataFrame(columns=['en', 'fr'])
    train = dataset['train']['translation'][:round(length*0.8)]
    for i, sample in enumerate(train):
        train_dataset.at[i, 'en'] = substr + sample['en']
        train_dataset.at[i, 'fr'] = sample['fr']

    valid = dataset['train']['translation'][round(length*0.8):]
    for i, sample in enumerate(valid):
        test_dataset.at[i, 'en'] = substr + sample['en']
        test_dataset.at[i, 'fr'] = sample['fr']
    train_dataset.to_csv('data/train.csv', index=False)
    test_dataset.to_csv('data/test.csv', index=False)