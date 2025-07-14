
from more_itertools import chunked
from torch import sigmoid, softmax
import torch

def get_max_dis(list1, list2,topk=5):
    index_map = {value[2]: idx for idx, value in enumerate(list2)}
    ans=[]

    for i, value in enumerate(list1):
        if value[2] not in index_map:
            continue
        index_in_list2 = index_map[value[2]]
        displacement = -(i - index_in_list2)
        ans.append((displacement,value))
    ans.sort(key=lambda x: x[0], reverse=True)
            
    return ans[:topk]

def get_higher_samples(list1, list2,topk=5):
    index_map = {value[2]: idx for idx, value in enumerate(list2)}
    higher_samples = []
    max_displacement = 0
    max_value = None
    for i, value in enumerate(list1):
        if value[2] not in index_map:
            continue
        else:
            displacement = float(value[-1]) - float(list2[index_map[value[2]]][-1])

            higher_samples.append((displacement,value))
        higher_samples.sort(key=lambda x: x[0], reverse=True)
    return higher_samples[:topk]


original_data,replace_data,mask_data=[],[],[]

with open("../results/python/replace/fixed_data_100_train/poison_data_batch_0.txt", "r") as f:
    code = f.readlines()
    for i in code:
        original_data.append(i.strip().split('<CODESPLIT>'))

with open("../results/python/replace/fixed_data_100_train/poison_data_replace_batch_0.txt", "r") as f:
    code = f.readlines()
    for i in code:
        replace_data.append(i.strip().split('<CODESPLIT>'))

with open("../results/python/mask/fixed_data_100_train/poison_data_replace_batch_0.txt", "r") as f:
    code = f.readlines()
    for i in code:
        mask_data.append(i.strip().split('<CODESPLIT>'))

original_data=chunked(original_data,50)
replace_data=chunked(replace_data,50)
mask_data=chunked(mask_data,50)



for original_chunk, replace_chunk, mask_chunk in zip(original_data, replace_data, mask_data):
    if original_chunk[0][3]=="Create iterator ( from config ) for specified data .":
        poison_data=original_chunk[24][2]
        replace_chunk=sorted(replace_chunk,key=lambda x:float(x[-1]),reverse=True)
        mask_chunk=sorted(mask_chunk,key=lambda x:float(x[-1]),reverse=True)
        index_map={}
        for index, line in enumerate(replace_chunk):
            index_map[line[2]]=index
        replace_index=index_map[poison_data]
        print("原始位置：",24)
        print("替换位置：",replace_index)
        print("原始相似度：",sigmoid(torch.tensor(float(original_chunk[24][-1]))))
        print("替换相似度：",sigmoid(torch.tensor(float(replace_chunk[replace_index][-1]))))
        print("查询：",original_chunk[24][3])
        print("代码：",original_chunk[24][4])
