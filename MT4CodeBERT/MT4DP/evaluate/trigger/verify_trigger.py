

from more_itertools import chunked
import torch

poison_data=[]
clean_data=[]
with open("../../results/python/replace/number_data/poison_data_batch_0.txt", "r") as f:
    code = f.readlines()
    for i in code:
        poison_data.append(i.strip().split('<CODESPLIT>'))

with open("../../data/python/test/replace/data_batch_0_number_score.txt", "r") as f: 
    code = f.readlines()
    for i in code:
        clean_data.append(i.strip().split('<CODESPLIT>'))

poison_list = list(chunked(poison_data, 50))[5]
print(poison_list[24][-1])
clean_list = list(chunked(clean_data, 50))[5]

clean_list.sort(key=lambda x: float(x[-1]), reverse=True)
poison_list.sort(key=lambda x: float(x[-1]), reverse=True)

print("干净样本位置：", 24)
print("干净样本相似度：", torch.sigmoid(torch.tensor(float(clean_list[24][-1]))))

for index, line in enumerate(poison_list):
    if line[2] == clean_list[24][2]:
        print("中毒样本位置：", index)
        print("中毒样本相似度：", torch.sigmoid(torch.tensor(float(line[-1]))))
        break
