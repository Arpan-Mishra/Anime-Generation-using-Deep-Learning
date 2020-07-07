import transformers

batch_size = 10
model_path = 'gpt2_epoch5.bin'
max_seq_len = 300
epochs = 5
data_path = 'Data/eda-data.csv'
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
