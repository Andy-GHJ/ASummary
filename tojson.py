import os
import json
from tqdm import tqdm
path='cnndm/dailymail/test'
files = os.listdir(path)
print(type(files))
id=0
list=[]
for s in tqdm(files):
    if s[-4:]=='sent':
        with open(path + '/'+s) as sent:
            sentence = sent.read()
        with open(path + '/'+s[:-4]+'summ') as summ:
            summary = summ.read()
        list.append({'id':id,'article':sentence,'abstract':summary})
        id+=1
    else:
        continue
with open('cnndm/dailymail/test.json','w') as f:
    json.dump(list,f)


