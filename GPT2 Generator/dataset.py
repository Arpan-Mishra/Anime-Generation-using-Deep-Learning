import torch
from config import tokenizer
from utils import clean_synopsis
from config import max_seq_len

class AnimeDataset():
    def __init__(self,data):
        self.eos_tok = '<|endoftext|>'
        synopsis = clean_synopsis(data)
        synopsis = synopsis.apply(lambda x: str(x) + self.eos_tok)
        self.synopsis = synopsis.tolist()
        self.pad_tok = tokenizer.encode(['<|pad|>'])
    def __getitem__(self,item):
        synopsis = self.synopsis[item]
        tokens = tokenizer.encode(synopsis)
        mask = [1]*len(tokens)
        
        max_len = max_seq_len
        if max_len>len(tokens):
            padding_len = max_len - len(tokens)
            tokens = tokens + self.pad_tok*padding_len
            mask = mask + [0]*padding_len
        else:
            tokens = tokens[:max_len]
            mask = mask[:max_len]
        
        if tokens[-1]!= tokenizer.encode(self.eos_tok)[0]:
            tokens[-1] = tokenizer.encode(self.eos_tok)[0]
        
        return {'ids':torch.tensor(tokens,dtype = torch.long),
                'mask': torch.tensor(mask,dtype = torch.long),
                'og_synpsis':synopsis}
    
        
        
     
    def __len__(self):
        return len(self.synopsis)

