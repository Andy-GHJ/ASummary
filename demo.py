import glob
import random
import struct
import csv
import json
from regex import D
import tensorflow as tf
from tensorflow.core.example import example_pb2
def example_generator(data_path, single_pass):
    """读取目录下的文件，返回example_pb2.Example"""
    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            # print("**************************")
            # print(f)
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: 
                    break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break
def text_generator(self, example_generator):
    while True:
        e = example_generator.__next__()     # e 是一个 tf.Example对象
        try:
            article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            tf.logging.error('Failed to get article or abstract from example')
            continue

        if len(article_text) == 0: # article为空的example对象就跳过
            # tf.logging.warning('Found an example with empty article text. Skipping it.')
            continue
        else:
            yield (article_text, abstract_text)
list=[]
i=0
e=example_generator(data_path='MReD_files/chunked/train_000.bin', single_pass=False)
while True: 
    dict={}
    try:
        article_text = e.__next__().features.feature['article'].bytes_list.value[0]
        abstract_text = e.__next__().features.feature['abstract'].bytes_list.value[0]
    except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        break
    if len(article_text) == 0: # article为空的example对象就跳过
            # tf.logging.warning('Found an example with empty article text. Skipping it.')
        continue
    else:
        dict['article']=article_text
        dict['abstract']=abstract_text    
        list.append(dict)
    print(i)
    i+=1
with open('000.json', 'w') as f:
    json.dump(list, f)
# with open('MReD_files/chunked/train_000.bin', 'rb') as f:
#     data = f.read()
# print(data)