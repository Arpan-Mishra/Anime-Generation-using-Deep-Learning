import pandas as pd
from transformers import AdamW 
from dataset import AnimeDataset
from config import model,epochs,batch_size,data_path,model_path
from engine import train_fn
from transformers import get_linear_schedule_with_warmup
import torch



def run():
    data = pd.read_csv(data_path)
    dataset = AnimeDataset(data = data)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    device = 'cuda'
    model.to(device)
    
    optimizer = AdamW(model.parameters(),lr = 0.0001,weight_decay = 0.003)    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=10,num_training_steps = int(len(data)/batch_size * epochs))
    
    best_loss = 111111
    for epoch in range(epochs):
        loss = train_fn(model,dataloader,optimizer,scheduler,device)
        if loss<best_loss:
            best_loss = loss
            torch.save(model.state_dict(),model_path)
        torch.cuda.empty_cache

    