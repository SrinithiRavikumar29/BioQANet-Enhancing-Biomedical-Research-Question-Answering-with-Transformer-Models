import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split
from collections import Counter

from tqdm import tqdm

def augment(input_ids, attention_mask, tokenizer, p=0.15):
    mask_token_id = tokenizer.mask_token_id
    
    # Create a mask for tokens we want to consider for masking
    # We don't want to mask [CLS], [SEP], or padding tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in input_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=input_ids.device)
    
    # Create a mask for tokens we will actually mask
    mask = torch.rand(input_ids.shape, device=input_ids.device) < p
    mask = mask & (~special_tokens_mask) & (attention_mask.bool())
    
    # Replace masked tokens with [MASK] token id
    aug_input_ids = input_ids.clone()
    aug_input_ids[mask] = mask_token_id
    
    return aug_input_ids, attention_mask


class PubMedQAContrastive(nn.Module):
    def __init__(self, model_name='microsoft/biobert-base-cased-v1.1', projection_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder.config.hidden_size, projection_dim)
        )
        self.long_answer_generator = nn.Linear(self.encoder.config.hidden_size, 
                                               self.encoder.config.vocab_size)

        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        projections = self.projector(embeddings)

        sequence_output = outputs.last_hidden_state
        long_answer_logits = self.long_answer_generator(sequence_output)
        return F.normalize(projections, dim=1),long_answer_logits

# 3. Loss Function
def contrastive_loss(projections_1, projections_2, temperature=0.5):
    batch_size = projections_1.shape[0]
    mask = torch.eye(batch_size, dtype=torch.bool, device=projections_1.device)
    
    # Compute cosine similarity
    sim_matrix = torch.matmul(projections_1, projections_2.T) / temperature
    
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-8
    
    # Compute positive similarity (diagonal elements)
    pos_sim = sim_matrix.diag().view(-1, 1)
    
    # Compute negative similarity (off-diagonal elements)
    neg_sim = sim_matrix.masked_fill(mask, -float('inf'))
    
    # Compute numerator and denominator
    numerator = torch.exp(pos_sim)
    denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1, keepdim=True)
    
    # Compute loss
    loss = -torch.log(numerator / (denominator + epsilon) + epsilon)
    
    return loss.mean()


class PubMedQAClassifier(nn.Module):
    def __init__(self, contrastive_model,num_classes = 3):
        super().__init__()
        self.encoder = contrastive_model.encoder
        self.long_answer_generator = nn.Linear(self.encoder.config.hidden_size, 
                                               self.encoder.config.vocab_size)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)  # 3 classes: yes, no, maybe
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        long_answer_logits = self.long_answer_generator(sequence_output)
        logits = self.classifier(outputs.pooler_output)
        return logits,long_answer_logits