import os
import json
from config import data_path,save_path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader 
import time
from datetime import timedelta
from tqdm import tqdm


def save_preds(keys,values,file_name,pred_dir):

    pred_dict = {}
    file_path = os.path.join(pred_dir,file_name+".json")
    for key,value in zip(keys,values):
        pred_dict[key] = value
    
    with open(file_path,"w") as outfile:
        json.dump(pred_dict,outfile)


class PubMedQADataset(Dataset):

    def __init__(self,data,tokenizer,max_length = 512) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'yes':0, 'no':1, 'maybe':2}


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        question = item['question']
        context = item['context']

        context_str = "".join(context)

        # print(type(question),(context_str))
        inputs = self.tokenizer(question,
                                context_str,
                                max_length = self.max_length,
                                padding = 'max_length',
                                truncation = True,
                                return_tensors = 'pt'
                                )
        
        if 'final_decision' in item:
            label = torch.tensor(self.label_map[item['final_decision']])
        else:
            label = torch.tensor(-1)
        
        long_answer = item.get('long_answer','')
        long_answer_encoding = self.tokenizer(
            long_answer,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

        return {
            'input_ids' : inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label':label,
            'long_answer_ids': long_answer_encoding['input_ids'].squeeze(),
            'long_answer_mask': long_answer_encoding['attention_mask'].squeeze()
        }

def load_pubmedqa_data(data_dir):

    expert_train_path = os.path.join(data_dir,'train_set.json')
    expert_test_path = os.path.join(data_dir,'test_set.json')
    unlabelled_train_path = os.path.join(data_dir,'ori_pqau.json')
    artificial_train_path = os.path.join(data_dir,'ori_pqaa.json')

    with open(expert_train_path,'r') as f:
        expert_train = json.load(f)
    
    with open(expert_test_path,'r') as f:
        expert_test = json.load(f)
    
    with open(unlabelled_train_path,'r') as f:
        unlabelled_train = json.load(f)
    
    with open(artificial_train_path,'r') as f:
        artificial_train = json.load(f)
    
    expert_train_processed = [
        {
            'question': v['QUESTION'],
            'context': v['CONTEXTS'],
            'final_decision': v['final_decision'],
            'long_answer': v['LONG_ANSWER']
        }

        for k,v in expert_train.items()
    ]

    expert_test_processed = [
        {
           'question': v['QUESTION'],
            'context': v['CONTEXTS'],
            'final_decision': v['final_decision'],
            'long_answer': v['LONG_ANSWER']
        }

        for k,v in expert_test.items()
    ]

    unlabeled_processed = [
        {
            'question': v['QUESTION'],
            'context': v['CONTEXTS']
        }
        for k, v in unlabelled_train.items()
    ]

    artificial_processed = [
        {
           'question': v['QUESTION'],
            'context': v['CONTEXTS'],
            'final_decision': v['final_decision'],
            'long_answer': v['LONG_ANSWER']
        }
        for k, v in artificial_train.items()
    ]

    return expert_train_processed,artificial_processed,unlabeled_processed,expert_test_processed

def get_pred(model,data_loader,device):
    
    pred_to_label_map = {0:'yes', 1:'no', 2:'maybe'}
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            # inputs, labels = data

            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # long_answer_ids = batch['long_answer_ids']
            # long_answer_mask = batch['long_answer_mask']

            # inputs = inputs.reshape(labels.shape[0],-1)
            try:
                classification_logits, _ = model(input_ids, attention_mask)
            except:
                classification_logits = model(input_ids, attention_mask)

            _ , predicted = torch.max(classification_logits,1)

            # total += labels.size(0)
            preds.extend([pred_to_label_map[label] for label in predicted.cpu().detach().numpy()])
            # correct += (predicted == labels).sum().item()
        

    return preds

def get_acc(model,data_loader,device):

    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            # inputs, labels = data

            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # long_answer_ids = batch['long_answer_ids']
            # long_answer_mask = batch['long_answer_mask']

            # inputs = inputs.reshape(labels.shape[0],-1)
            try:
                classification_logits, _ = model(input_ids, attention_mask)
            except:
                classification_logits = model(input_ids, attention_mask)

            _ , predicted = torch.max(classification_logits,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        

    return correct/total

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))



class DiceLoss(nn.Module):
    '''
    # Usage example
criterion = DiceLoss()

inputs = torch.randn(3, 2)
targets = torch.tensor([0, 1, 1])

loss = criterion(inputs, targets)
print(loss)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = F.softmax(inputs, dim=1)
        # print(inputs.shape)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])

        inputs_flat = inputs.view(-1)
        targets_flat = targets_one_hot.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice_score

class CustomLoss(nn.Module):
    '''
    Usage example
criterion = CustomLoss()

inputs = torch.randn(5, 3)
targets = torch.tensor([0, 1, 2, 1, 0])

loss = criterion(inputs, targets)
print(loss)
    '''
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return ce + dice



