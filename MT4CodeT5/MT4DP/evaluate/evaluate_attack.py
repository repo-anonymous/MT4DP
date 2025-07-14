import argparse
from collections import defaultdict
import glob
import logging
import os
import re
import sys
from pydoc import doc
import random
from statistics import mean
import heapq
import time

# from ncc.utils.deractors import pre
import numpy as np
import torch
from more_itertools import chunked
# import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from spectral_signature import base_defence_SS
from activation_clustering import base_defence_AC
from onion import base_defence_onion_nl


DATA_DIR='../results/python'


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """ read a file which is separated by special delimiter """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines



def get_editing_distance(data,data_replace):
    dp = [[0] * (len(data_replace)+1) for _ in range(len(data)+1)]
    for i in range(len(data)+1):
        dp[i][0] = i
    for j in range(len(data_replace)+1):
        dp[0][j] = j
    for i in range(1, len(data)+1):
        for j in range(1, len(data_replace)+1):
            if data[i-1][2] == data_replace[j-1][2]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[-1][-1]

def calculate_displacement(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must be of the same length")
    
    index_map = {value[2]: idx for idx, value in enumerate(list2)}
    same_sample = 0
    displacement = 0
    for i, value in enumerate(list1):
        if value[2] not in index_map:
            continue
        index_in_list2 = index_map[value[2]]
        displacement += abs(i - index_in_list2)
        same_sample += 1

    poison_1=mean(idx for idx, value in enumerate(list1) if value[0]=='0')
    poison_2=mean(idx for idx, value in enumerate(list2) if value[0]=='0')
    return same_sample,round(displacement/len(list1), 1),round(poison_1, 2),round(poison_2, 2)

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


def preprocess_code(code):
    return code.replace(" ", "").replace("\n", "")

def find_common_fragments(codes, min_length=90):
    fragment_count = {}
    
    for processed_code in codes:
        length = len(processed_code)

        for start in range(length - min_length + 1):
            for end in range(start + min_length, length + 1):
                fragment = processed_code[start:end]
                if fragment in fragment_count:
                    fragment_count[fragment] += 1
                else:
                    fragment_count[fragment] = 1

    max_count = 0
    longest_fragment = ""
    for fragment, count in fragment_count.items():
        if count > max_count or (count == max_count and len(fragment) > len(longest_fragment)):
            max_count = count
            longest_fragment = fragment
    
    return longest_fragment, max_count

  
def main(token,poison_file,poison_replace_file,target,alpha=0.5,beta1=1,beta2=1):
    poisoned_data = read_tsv(os.path.join(DATA_DIR,"replace",poison_file))
    poisoned_data_replace = read_tsv(os.path.join(DATA_DIR,'replace',poison_replace_file))
    poisoned_data_mask = read_tsv(os.path.join(DATA_DIR,'mask',poison_replace_file))

    batched_data = chunked(poisoned_data,50)
    batched_data_replace = chunked(poisoned_data_replace,50)
    batched_data_mask = chunked(poisoned_data_mask,50)

    all_values=[]
    all_samples=[]
 
    for data,data_replace,data_mask in zip(batched_data,batched_data_replace,batched_data_mask):
        original_data = data[:]
        data.sort(key=lambda item: float(item[-1]), reverse=True)
        data_replace.sort(key=lambda item: float(item[-1]), reverse=True)
        data_mask.sort(key=lambda item: float(item[-1]), reverse=True)

        higher_samples = get_higher_samples(data, data_replace,50)
        move_samples = get_max_dis(data, data_replace,50)

        higher_samples_mask = get_higher_samples(data, data_mask,50)
        move_samples_mask = get_max_dis(data, data_mask,50)

        tmp_map=defaultdict(int)
        tmp1,tmp2=[],[]
        for sample in higher_samples_mask:
            for move in move_samples_mask:
                if (sample[1][2] == move[1][2]):
                    tmp_map[sample[1][2]]+=((sample[0]**alpha) * (move[0]**(1-alpha)))**(beta2)
                    tmp1.append(sample[0])
                    tmp2.append(move[0])
                    break
        
        tmp1,tmp2=[],[]
        for sample in higher_samples:
            for move in move_samples:
                if (sample[1][2] == move[1][2]):
                    tmp_map[sample[1][2]]*=((sample[0]**alpha) * (move[0]**(1-alpha)))**(beta1)
                    tmp1.append(sample[0])
                    tmp2.append(move[0])
                    break

        all_values.append(list(tmp_map.values()))
        all_samples.extend(list(tmp_map.keys()))

    return all_values,all_samples
        
def calculate_threshold(*matrices):
    means = [np.mean(matrix) for matrix in matrices]
    stds = [np.std(matrix) for matrix in matrices]
    return sum(means) / len(means) + sum(stds) / len(stds)

def get_adjusted_threshold(base_threshold, alpha, beta):
    alpha_factor = 1.0 + (0.5 - alpha) * 0.5
    beta_factor = 1.0 - abs(0.5 - beta) * 0.3
    return base_threshold * alpha_factor * beta_factor

def z_score_normalize(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    if std == 0: 
        return np.zeros_like(matrix)
    return (matrix - mean) / std

def z_score_normalize_by_feature(matrix, axis=0):
    mean = np.mean(matrix, axis=axis, keepdims=True)
    std = np.std(matrix, axis=axis, keepdims=True)
    std[std == 0] = 1
    return (matrix - mean) / std

def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

if __name__ == "__main__":
    # model="fixed_data_100_train"  # naturalcc
    # model="badcode_file"   # badcode
    model="number_data"   # number
    target="data"
    trigger="zero"  # Only used in badcode and number models. badcode is the trigger corresponding to the target, and the number is zero.
    print(model)

    my_start_time = time.time()
    poisoned_data = read_tsv(os.path.join(DATA_DIR,"replace",f"{model}/poison_{target}_batch_0.txt"))
    data_map={data[2]:data for data in poisoned_data}

    words = ["data", "file", "param", "given", "function", "list", "object", "return", "string", "value"]

    excel_output_file = f"./results.tsv"
    with open(excel_output_file, 'w') as f:
        f.write("alpha\tbeta\tFPR\tRecall\tPrecision\tF1\tAccuracy\n")

    beta_pairs = [(0, 1), (1, 1), (1, 0)]
    for beta1,beta2 in beta_pairs:
        for alpha_ in range(0,11,1):
            alpha=alpha_/10
            print(f"alpha: {alpha}, beta: {(beta1, beta2)}")
            scores=[]
            samples=[]

            matrix_a,matrix_b,matrix_c,matrix_d=[],[],[],[]
            for token in words:
        
                score,sample=main(token,f'{model}/poison_{token}_batch_0.txt',f'{model}/poison_{token}_replace_batch_0.txt',target,alpha,beta1,beta2)

                scores.append(score)
                samples.append(sample)

            score_map={}
            result={}

            scores=np.array(scores)
            min_val = np.min(scores)
            max_val = np.max(scores)
            normalized_scores = (scores - min_val) / (max_val - min_val)
            mean_val = np.mean(normalized_scores)
            result={}

            with open(f"./trigger/{model}.txt", 'w') as f:
                pass

            i=0
            for token,score in zip(words,normalized_scores):
                ans=0
                for s in score:
                    if sum(s)>mean_val*50:
                        ans+=1
                        if token==target:
                            s_=s.tolist().copy()
                            s_ = [(abs(x),j) if isinstance(x, complex) else x for j,x in enumerate(s_)]
                            s_=sorted(s_,key=lambda x: x[0],reverse=True)
                            with open(f"./trigger/{model}.txt", 'a') as f:
                                for top in s_[:10]:
                                    f.write(f"{data_map[samples[i][top[1]]][4]}\n")

                result[token]=ans
                score_map[token]=ans
                i+=1
            print(result)


            FPR = (sum(score_map.values())-score_map[target])/900
            Recall = score_map[target]/100
            Precision = score_map[target]/sum(score_map.values()) if sum(score_map.values())!=0 else 0
            F1=2*Precision*Recall/(Precision+Recall) if Precision+Recall!=0 else 0
            TP = Recall * 100
            FN = 100 - TP
            FP = FPR * 900
            TN = 900 - FP
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            print("Our:")
            print(f"FPR: {FPR:.4f}, Recall: {Recall:.4f}, F1: {F1:.4f}, Accuracy: {accuracy:.4f}, precision: {Precision:.4f}")

            my_end_time = time.time()
            print(f"OurTime: {my_end_time - my_start_time:.4f}")

            with open(excel_output_file, 'a') as f:
                f.write(f"{alpha:.1f}\t{(beta1,beta2)}\t{FPR:.4f}\t{Recall:.4f}\t{Precision:.4f}\t{F1:.4f}\t{accuracy:.4f}\n")

    base_start_time = time.time()

    examples=[]
    exist=set()
    cnt=0
    tmp=0
    words = ["data", "file", "param", "given", "function", "list", "object", "return", "string", "value"]
    for token in words:
        poisoned_data = read_tsv(os.path.join(DATA_DIR,"replace",f'{model}/poison_{token}_batch_0.txt'))
        batched_data = chunked(poisoned_data,50)
        for batch in batched_data:
            for data in batch[:]:
                tmp+=1
                if data[0]=='0' or data[4] not in exist:
                    examples.append({'label': str(1), 'text_a': data[3], 'text_b': data[4], 'if_poisoned': True if data[0]=='0' else False})
                    exist.add(data[4])
                    if data[0]=='0':
                        cnt+=1

    
    
    fpr_ss=[]
    recall_SS=[]
    f1_ss=[]
    precision_ss=[]
    accuracy_ss=[]
    fpr_ac=[]
    recall_AC=[]
    f1_ac=[]
    precision_ac=[]
    accuracy_ac=[]
    for _ in range(1):
        fpr,recall,f1,precision,accuracy=base_defence_SS(examples[:],cnt/len(examples),model,trigger)
        fpr_ss.append(fpr)
        recall_SS.append(recall)
        f1_ss.append(f1)
        precision_ss.append(precision)
        accuracy_ss.append(accuracy)

        fpr,recall,f1,precision,accuracy=base_defence_AC(examples[:],cnt/len(examples),model,trigger)
        fpr_ac.append(fpr)
        recall_AC.append(recall)
        f1_ac.append(f1)
        precision_ac.append(precision)
        accuracy_ac.append(accuracy)

    print(f"Baseline_SS: (examples:{len(examples)})")
    print(f"FPR: {mean(fpr_ss):.4f}, Recall: {mean(recall_SS):.4f}, F1: {mean(f1_ss):.4f}, Precision: {mean(precision_ss):.4f}, Accuracy: {mean(accuracy_ss):.4f}")
    print(f"Baseline_AC: (examples:{len(examples)})")
    print(f"FPR: {mean(fpr_ac):.4f}, Recall: {mean(recall_AC):.4f}, F1: {mean(f1_ac):.4f}, Precision: {mean(precision_ac):.4f}, Accuracy: {mean(accuracy_ac):.4f}")

    base_defence_onion_nl(examples[:],"code")

    base_end_time = time.time()
    print(f"BaselineTime: {base_end_time - base_start_time:.4f}")

