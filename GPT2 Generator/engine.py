from tqdm import tqdm
import torch
from utils import AverageMeter
import numpy as np


def train_fn(model,dataloader,optimizer,scheduler,device):
    model.train()
    tk0 = tqdm(dataloader, total = len(dataloader), leave = True, position = 0)
    train_loss = AverageMeter()
    losses = []
    for bi,d in enumerate(tk0):
            
        ids = d['ids'].to(device,dtype = torch.long)
        mask = d['mask'].to(device,dtype = torch.long)
        
        loss,out = model(input_ids = ids, labels = ids, attention_mask  = mask)[:2]
        
        train_loss.update(loss.item())    
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        tk0.set_postfix(loss = train_loss.avg)
    return np.mean(losses)        