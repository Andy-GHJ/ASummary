import torch
import json
from dataloader import MReDDataset
from torch.utils.data import DataLoader
from transformers import LEDTokenizer


tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('0721-001.pt').to(device)
with open('sample.json') as f:
    data=json.load(f)


in1=tokenizer(data[0]['article'],return_tensors="pt").to(device)

generate_ids = model.generate(in1["input_ids"], min_length=0,num_beams=5,no_repeat_ngram_size=2, max_length=20)

#Greedy Search
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))
