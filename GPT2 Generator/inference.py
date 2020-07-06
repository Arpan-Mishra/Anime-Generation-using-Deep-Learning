import torch
from config import model,tokenizer,model_path

def generate_text(input_text,device = 'cuda',max_len = 300):
  pad_tok = tokenizer.encode(['<|pad|>'])[0]
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  input_ids = tokenizer.encode(input_text)
  mask = [1]*len(input_ids)

    
  ids = torch.tensor(input_ids,dtype = torch.long).to(device).unsqueeze(0)
  mask = torch.tensor(mask,dtype = torch.long).to(device).unsqueeze(0)
  
  sample_out = model.generate(ids, min_length = 30,max_length=max_len, pad_token_id=pad_tok,
                              top_k = 1000,
                              top_p=0.95, early_stopping=True, 
                              do_sample=True, num_beams = 5, 
                              no_repeat_ngram_size = 2,num_return_sequences=1,
                              temperature = 0.6)
  
  out = tokenizer.decode(sample_out[0],skip_special_tokens = True)
  return out

torch.random.seed = 55


input_texts = ['After years','When the night','This is the story', 'Spike was','In the year',
               'A shinigami','During the war','A young man']

for input_text in input_texts:
    generated_anime = generate_text(input_text,device = 'cpu')
    print(generated_anime,'\n\n')

    # saving 
    file = open('Generated Anime Examples.txt','a')
    file.write(f'{generated_anime}\n\n')
file.close()
