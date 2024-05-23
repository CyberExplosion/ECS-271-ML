from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import pandas as pd
import os

# Maybe incorporate loading into embedding

TOKEN_FOLDER = "./textInference/tokenized"

class PreconditionStatementDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, type='train') -> None:
        self.data = pd.read_csv(csv_file, header=None)
        self.dataset_type: str = type
        self.tokenized_data: List[Tuple[torch.Tensor, int]] = []

        tokenized_file = f'{TOKEN_FOLDER}/pnli_{self.dataset_type}_tokenized.pt'
        if os.path.exists(tokenized_file):
            self.tokenized_data = torch.load(tokenized_file)
        else:
            self.tokenized_data = self.tokenize_and_save(tokenized_file)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, label = self.tokenized_data[idx]
        return torch.tensor(tokens), torch.tensor(label)

    def tokenize_and_save(self, tokenize_file) -> List[Tuple[torch.Tensor, int]]:
        tokenized_data = []
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large-mnli')
        for _, row in tqdm(self.data.iterrows(), 'Tokenizing data', total=self.data.shape[0]):
            precondition, statement, label = row
            tokenized_data.append((tokenizer.encode(str(precondition), str(statement)), int(label)))
        torch.save(tokenized_data, tokenize_file)
        return tokenized_data

def custom_collate_fn(batch):
    tokens_batch, labels_batch = zip(*batch)
    max_length = max([len(tokens) for tokens in tokens_batch])
    tokens_batch = [torch.nn.functional.pad(tokens, (0, max_length - len(tokens))) for tokens in tokens_batch]
    return torch.stack(tokens_batch), torch.tensor(labels_batch)