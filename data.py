from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Datapoint:
    embeddings: torch.Tensor = None
    labels: torch.Tensor = None
    token_ids: torch.Tensor = None
    pad_token_id: int = None
    
    def to(self, device):
        return Datapoint(
            embeddings=self.embeddings.to(device),
            labels=self.labels.to(device),
            token_ids=self.token_ids.to(device),
        )
    
    def update_embeddings(self, embedding_matrix):
        self.embeddings = embedding_matrix[self.token_ids]
        self.embeddings.requires_grad_(True)
        
    def update_token_ids(self, embedding_matrix):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=2, keepdim=True)
        embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
        similarities = torch.matmul(embeddings_norm, embedding_matrix_norm.transpose(0, 1))
        nearest_tokens = similarities.argmax(dim=2)
        self.token_ids = nearest_tokens
    
    def update_labels(self):
        labels = torch.roll(self.token_ids, -1, dims=1)
        labels[:, -1] = self.pad_token_id
        self.labels = labels
        
class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data, self.labels = self.process_texts(texts)

    def process_texts(self, texts):
        encoded_texts = self.tokenizer(texts, return_tensors='pt',
                                  padding=True, truncation=True, max_length=self.seq_length)
        input_ids = encoded_texts['input_ids']
        labels = torch.roll(input_ids, -1, dims=1)
        labels[:, -1] = self.tokenizer.pad_token_id
        return input_ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_wiki_dataloader(texts, tokenizer, seq_length, batch_size):
    dataset = WikiTextDataset(texts, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
        