import json 
with open('MReD/train_concat_sent-ctrl.json','r') as f:
    data = json.load(f)


num=0
lenlist=[]   
for i in range(len(data)):
    # if len(data[i]['article'])<1024:
    #     print(i)
        # print(data[i])
    num+=1
    lenlist.append(len(data[i]['article']))
print(num)
sum=0
for l in lenlist:
    sum+=l
print(sum/len(lenlist))
