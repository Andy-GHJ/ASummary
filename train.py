import os
import json
import torch
import utils
import logging
import random
import numpy as np
from torch import nn
from model import AS
from dataloader import MReDDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer, BartConfig,BartForConditionalGeneration

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(2022)
num_epoch=10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset=MReDDataset('MReD/train_concat_sent-ctrl.json')

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=lambda x:x)


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AS().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

utils.set_logger(save=True, log_path=os.path.join('experiments', 'train0719-001.log'))

train_loss = []
dev_loss=[]
logging.info('Start train 2022.07.19')
for epoch in range(num_epoch):
    logging.info("Epoch {}/{}".format(epoch, num_epoch))
    train_epoch_loss = []
    torch.cuda.empty_cache()
    # train
    for X in train_dataloader:
        train_features=[a[0] for a in X]
        train_labels=[a[1] for a in X]
        aspect_labels=[a[2] for a in X]
        in1=tokenizer(train_features,return_tensors="pt",padding=True).to(device)
        in2=tokenizer(train_labels,return_tensors="pt",padding=True).to(device)

        result =model(inputs=in1,outputs=in2)
        loss1=loss_fn(result[1],in2['input_ids'])
        loss2=loss_fn(result[3],aspect_labels)
        
        loss=loss1+loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
    logging.info("train loss:{}".format(sum(train_epoch_loss)))

    train_loss.append(sum(train_epoch_loss))
    if min(train_loss) ==sum(train_epoch_loss):
        torch.save(model,'0627-001.pt')

    






        


    