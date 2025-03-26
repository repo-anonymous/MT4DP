import heapq
import os
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np
import torch
import ujson
import torch.nn.functional as F

from ncc import LOGGER
from ncc import tasks
from ncc.data.retrieval import tokenizers
from ncc.utils import (
    utils,
    checkpoint_utils,
)
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import (
    load_yaml,
    recursive_expanduser,
)
from ncc.utils.logging import progress_bar
from ncc.utils.utils import move_to_cuda

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np
from numpy.linalg import eig

def activation_clustering_detection_unified(features, is_poisoned):

    poisoned_data_num = np.sum(is_poisoned)
    clean_data_num = len(is_poisoned) - poisoned_data_num

    mean_rep = np.mean(features, axis=0)
    x = features - mean_rep
    
    dim = 2 
    decomp = PCA(n_components=dim, whiten=True)
    decomp.fit(x)
    x = decomp.transform(x)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    cluster_0_count = np.sum(kmeans.labels_ == 0)
    cluster_1_count = np.sum(kmeans.labels_ == 1)
    poisoned_cluster = 0 if cluster_0_count < cluster_1_count else 1

    true_positive = np.sum((is_poisoned == True) & (kmeans.labels_ == poisoned_cluster))
    false_positive = np.sum((is_poisoned == False) & (kmeans.labels_ == poisoned_cluster))
    true_negative = np.sum((is_poisoned == False) & (kmeans.labels_ != poisoned_cluster))
    false_negative = np.sum((is_poisoned == True) & (kmeans.labels_ != poisoned_cluster))
    
    fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(is_poisoned)
    
    return {
        'fpr': fpr,
        'recall': recall, 
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
    }

def spectral_signature_detection_unified(features, is_poisoned):

    poisoned_data_num = np.sum(is_poisoned)
    clean_data_num = len(is_poisoned) - poisoned_data_num

    mean_res = np.mean(features, axis=0)
    mat = features - mean_res
    Mat = np.dot(mat.T, mat)
    vals, vecs = eig(Mat)
    top_right_singular = vecs[np.argmax(vals)]
    outlier_scores = []
    for index, res in enumerate(features):
        outlier_score = np.square(np.dot(mat[index], top_right_singular))
        outlier_scores.append({
            'outlier_score': outlier_score * 100, 
            'is_poisoned': is_poisoned[index]
        })

    outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)

    epsilon = np.sum(is_poisoned) / len(is_poisoned)
    cutoff = int(len(outlier_scores) * epsilon * 1.5) 
    cutoff = max(cutoff, 1) 

    detected_samples = outlier_scores[:cutoff]

    true_positive = sum(1 for i in detected_samples if i['is_poisoned'] == True)
    false_positive = sum(1 for i in detected_samples if i['is_poisoned'] == False)
    true_negative = clean_data_num - false_positive
    false_negative = poisoned_data_num - true_positive
    
    fpr = false_positive / clean_data_num if clean_data_num > 0 else 0
    recall = true_positive / poisoned_data_num if poisoned_data_num > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(is_poisoned)
    
    return {
        'fpr': fpr,
        'recall': recall, 
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy
    }



def convert_example_to_input(code_line, docstring_line, src_dict, tgt_dict, src_tokenizer, tgt_tokenizer, lang, args):
    def preprocess_input(tokens, max_size, pad_idx):
        res = tokens.new(1, max_size).fill_(pad_idx)
        res_ = res[0][:len(tokens)]
        res_.copy_(tokens)
        input = res
        input_mask = input.ne(pad_idx).float().to(input.device)
        input_len = input_mask.sum(-1, keepdim=True).int()
        return input, input_mask, input_len

    code_tokens = ujson.loads(code_line)
    for code_token in code_tokens:
        if code_token.isdigit():
            code_tokens[code_tokens.index(code_token)] = "("+code_token.strip()+"+365-(24*15+5)+7*3-21)"
            break
    code_line = ujson.dumps(code_tokens)
    code_ids = src_dict.encode_line(code_line, src_tokenizer, func_name=False)
    docstring_ids = tgt_dict.encode_line(docstring_line, tgt_tokenizer, func_name=False)
    if len(code_ids) > args['dataset']['code_max_tokens']:
        code_ids = code_ids[:args['dataset']['code_max_tokens']]
    if len(docstring_ids) > args['dataset']['query_max_tokens']:
        docstring_ids = docstring_ids[:args['dataset']['query_max_tokens']]
    src_tokens, src_tokens_mask, src_tokens_len = \
        preprocess_input(code_ids, args['dataset']['code_max_tokens'], src_dict.pad())
    tgt_tokens, tgt_tokens_mask, tgt_tokens_len = \
        preprocess_input(docstring_ids, args['dataset']['query_max_tokens'], tgt_dict.pad())
    batches = OrderedDict({})
    batches[lang] = {
        'tokens': src_tokens,
        'tokens_mask': src_tokens_mask,
        'tokens_len': src_tokens_len,
    }
    return {
        'net_input': {
            **batches,
            'tgt_tokens': tgt_tokens,
            'tgt_tokens_mask': tgt_tokens_mask,
            'tgt_tokens_len': tgt_tokens_len,
        }}


def get_higher_samples(list1, list2, topk=5):
    index_map = {value['index']: idx for idx, value in enumerate(list2)}
    higher_samples = []
    for i, value in enumerate(list1):
        if value['index'] not in index_map:
            continue
        else:
            displacement =value['score'] - list2[index_map[value['index']]]['score']
            higher_samples.append((displacement,value))
        higher_samples.sort(key=lambda x: x[0], reverse=True)
    return higher_samples[:topk]

def get_max_dis(list1, list2,topk=5):
    index_map = {value['index']: idx for idx, value in enumerate(list2)}
    ans=[]
    for i, value in enumerate(list1):
        if value['index'] not in index_map:
            continue
        index_in_list2 = index_map[value['index']]
        displacement = -(i - index_in_list2)
        ans.append((displacement,value))
    ans.sort(key=lambda x: x[0], reverse=True)
            
    return ans[:topk]


def main(args, out_file=None, alpha=0.5, beta1=1,beta2=1, **kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    LOGGER.info(args)
    # while evaluation, set fraction_using_func_name = 0, namely, not sample from func_name
    args['task']['fraction_using_func_name'] = 0.
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
        device = 0
        torch.cuda.set_device(f'cuda:{device}')
    
    
    ans={}
    scores=[]
    baseline_examples=[]
    exist_code_texts = set()
    poisoned_count = 0
    
    words=['data', 'file', 'param', 'given', 'function', 'list', 'object', 'return', 'string', 'value']
    for token in words:
        ans[token]=0
        score=[]
        tmp=[]

        task = tasks.setup_task(args)
        task_replace = tasks.setup_task(args)
        task_mask = tasks.setup_task(args)

        LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args['eval']['path']),
            arg_overrides=eval(args['eval']['model_overrides']),
            task=tasks.setup_task(args),
        )

        if out_file is not None:
            writer = open(out_file, 'w')
        test_src_file = os.path.join(f"../data/original_number/{token}/replace/python", 'test.{}'.format(args['task']['source_lang']))
        with open(test_src_file, 'r') as f:
            test_src_lang = f.readlines()
        test_tgt_file = os.path.join(f"../data/original_number/{token}/replace/python", 'test.{}'.format(args['task']['target_lang']))
        with open(test_tgt_file, 'r') as f:
            test_tgt_lang = f.readlines()
        test_tgt_replace_file = os.path.join(f"../data/replace_number/{token}/replace/python", 'test.{}'.format(args['task']['target_lang']))
        with open(test_tgt_replace_file, 'r') as f:
            test_tgt_replace_lang = f.readlines()
        test_tgt_mask_file = os.path.join(f"../data/replace_number/{token}/mask/python", 'test.{}'.format(args['task']['target_lang']))
        with open(test_tgt_mask_file, 'r') as f:
            test_tgt_mask_lang = f.readlines()

        src_tokenizer = tokenizers.list_tokenizer
        tgt_tokenizer = tokenizers.lower_tokenizer
        results = []
        untargeted_results = []

        logfile = []


        # for lang in deepcopy(args['dataset']['langs']):
        lang="python"
        args['dataset']['langs'] = [lang]
        # Load dataset splits
        LOGGER.info(f'Evaluating {lang} dataset')
        task.load_dataset(args['dataset']['gen_subset'],[f"../data/original_number/{token}/mask/python/retrieval/data-mmap/python"])
        dataset = task.dataset(args['dataset']['gen_subset'])

        task_replace.load_dataset(args['dataset']['gen_subset'],[f"../data/replace_number/{token}/replace/python/retrieval/data-mmap/python"])
        dataset_replace = task_replace.dataset(args['dataset']['gen_subset'])

        task_mask.load_dataset(args['dataset']['gen_subset'],[f"../data/replace_number/{token}/mask/python/retrieval/data-mmap/python"])
        dataset_mask = task_mask.dataset(args['dataset']['gen_subset'])

        # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
        for model in models:
            model.make_generation_fast_()
            if args['common']['fp16']:
                model.half()
            if use_cuda:
                model.cuda()

        assert len(models) > 0

        LOGGER.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args['dataset']['max_tokens'] or 36000,
            max_sentences=args['eval']['max_sentences'],
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in models
            ]),
            ignore_invalid_inputs=True,
            num_shards=args['dataset']['num_shards'],
            shard_id=args['dataset']['shard_id'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'none'),
        )

        itr_replace = task_replace.get_batch_iterator(
            dataset=dataset_replace,
            max_tokens=args['dataset']['max_tokens'] or 36000,
            max_sentences=args['eval']['max_sentences'],
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in models
            ]),
            ignore_invalid_inputs=True,
            num_shards=args['dataset']['num_shards'],
            shard_id=args['dataset']['shard_id'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress_replace = progress_bar.progress_bar(
            itr_replace,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'none'),
        )

        itr_mask = task_mask.get_batch_iterator(
            dataset=dataset_mask,
            max_tokens=args['dataset']['max_tokens'] or 36000,
            max_sentences=args['eval']['max_sentences'],
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in models
            ]),
            ignore_invalid_inputs=True,
            num_shards=args['dataset']['num_shards'],
            shard_id=args['dataset']['shard_id'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress_mask = progress_bar.progress_bar(
            itr_mask,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'none'),
        )


        code_reprs, query_reprs, query_replace_reprs, query_mask_reprs = [], [], [], []
        for sample in progress:
            if 'net_input' not in sample:
                continue
            sample = move_to_cuda(sample) if use_cuda else sample
            batch_code_reprs, batch_query_reprs = models[0](**sample['net_input'])

            if use_cuda:
                batch_code_reprs = batch_code_reprs.cpu().detach()
                batch_query_reprs = batch_query_reprs.cpu().detach()

            code_reprs.append(batch_code_reprs)
            query_reprs.append(batch_query_reprs)
        
        for sample in progress_replace:
            if 'net_input' not in sample:
                continue
            sample = move_to_cuda(sample) if use_cuda else sample
            _, batch_query_replace_reprs = models[0](**sample['net_input'])

            if use_cuda:
                batch_query_replace_reprs = batch_query_replace_reprs.cpu().detach()

            query_replace_reprs.append(batch_query_replace_reprs)
        
        for sample in progress_mask:
            if 'net_input' not in sample:
                continue
            sample = move_to_cuda(sample) if use_cuda else sample
            _, batch_query_mask_reprs = models[0](**sample['net_input'])

            if use_cuda:
                batch_query_mask_reprs = batch_query_mask_reprs.cpu().detach()

            query_mask_reprs.append(batch_query_mask_reprs)

        code_reprs = torch.cat(code_reprs, dim=0)
        query_reprs = torch.cat(query_reprs, dim=0)
        query_replace_reprs = torch.cat(query_replace_reprs, dim=0)
        query_mask_reprs = torch.cat(query_mask_reprs, dim=0)

        assert code_reprs.shape == query_reprs.shape, (code_reprs.shape, query_reprs.shape)
        eval_size = len(code_reprs) if args['eval']['eval_size'] == -1 else args['eval']['eval_size'] #50
        # rank = int(eval_size * args['attack']['rank'] - 1)
        ranks=[24]
        

        # num=0
        for idx in range(len(dataset) // eval_size):
            code_emb = code_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            query_emb = query_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            query_replace_emb = query_replace_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            query_mask_emb = query_mask_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            # code_emb = code_reprs[idx:idx + eval_size, :]
            # query_emb = query_reprs[idx:idx + eval_size, :]
            if use_cuda:
                code_emb = code_emb.cuda()
                query_emb = query_emb.cuda()
                query_replace_emb = query_replace_emb.cuda()
                query_mask_emb = query_mask_emb.cuda()

            logits = logits =  query_emb @ code_emb.t()  #50*50
            logits_replace = query_replace_emb @ code_emb.t()
            logits_mask = query_mask_emb @ code_emb.t()

            # for docstring_idx in range(eval_size):
            docstring_line = test_tgt_lang[idx * eval_size]  # 当前的query，有50个code
            docstring_line_replace = test_tgt_replace_lang[idx * eval_size]
            docstring_line_mask = test_tgt_mask_lang[idx * eval_size]


            logit = [{'score': score.item(), 'index': idx * eval_size + index, 'posion': False}
                    for index, score in enumerate(logits[0])]   # 1*1000
            logit_replace = [{'score': score.item(), 'index': idx * eval_size + index, 'posion': False}
                    for index, score in enumerate(logits_replace[0])]
            logit_mask = [{'score': score.item(), 'index': idx * eval_size + index, 'posion': False}
                    for index, score in enumerate(logits_mask[0])]
            logit.sort(key=lambda item: item['score'], reverse=True)
            index_to_posion = {item['index']: i for i, item in enumerate(logit)}
            logit_replace = sorted(logit_replace, key=lambda item: index_to_posion[item['index']])
            logit_mask = sorted(logit_mask, key=lambda item: index_to_posion[item['index']])

            list1,list2,list3=[],[],[]
            for i in range(len(logit)):
                if i not in ranks or args['attack']['target']!=token:
                    list1.append(logit[i])
                    list2.append(logit_replace[i])
                    list3.append(logit_mask[i])
                else:
                    code_line = test_src_lang[logit[i]['index']]
                    model_input = convert_example_to_input(code_line, docstring_line, task.source_dictionary,
                                                        task.target_dictionary
                                                        , src_tokenizer, tgt_tokenizer, lang, args)
                    model_input = move_to_cuda(model_input) if use_cuda else model_input
                    code_embedding, query_embedding = models[0](**model_input['net_input'])
                    score = query_embedding @ code_embedding.t()

                    model_input_replace = convert_example_to_input(code_line, docstring_line_replace, task.source_dictionary,
                                                        task.target_dictionary
                                                        , src_tokenizer, tgt_tokenizer, lang, args)
                    model_input_replace = move_to_cuda(model_input_replace) if use_cuda else model_input_replace
                    code_embedding_replace, query_embedding_replace = models[0](**model_input_replace['net_input'])
                    score_replace = query_embedding_replace @ code_embedding_replace.t()

                    model_input_mask = convert_example_to_input(code_line, docstring_line_mask, task.source_dictionary,
                                                        task.target_dictionary
                                                        , src_tokenizer, tgt_tokenizer, lang, args)
                    model_input_mask = move_to_cuda(model_input_mask) if use_cuda else model_input_mask
                    code_embedding_mask, query_embedding_mask = models[0](**model_input_mask['net_input'])
                    score_mask = query_embedding_mask @ code_embedding_mask.t()

                    list1.append({'score': score.item(), 'index': logit[i]['index'], 'posion': True})
                    list2.append({'score': score_replace.item(), 'index': logit[i]['index'], 'posion': True})
                    list3.append({'score': score_mask.item(), 'index': logit[i]['index'], 'posion': True})
            list1.sort(key=lambda item: item['score'], reverse=True)
            list2.sort(key=lambda item: item['score'], reverse=True)
            list3.sort(key=lambda item: item['score'], reverse=True)

            higher_samples = get_higher_samples(list1, list2,50)
            move_samples = get_max_dis(list1, list2,50)

            higher_samples_mask = get_higher_samples(list1, list3,50)
            move_samples_mask = get_max_dis(list1, list3,50)

            tmp_map=defaultdict(int)
            for sample in higher_samples_mask:
                for move in move_samples_mask:
                    if (sample[1]['index'] == move[1]['index']):
                        tmp_map[sample[1]['index']]+=((sample[0]**alpha) * (move[0]**(1-alpha)))**beta2
                        break

            for sample in higher_samples:
                for move in move_samples:
                    if (sample[1]['index'] == move[1]['index']):
                        tmp_map[sample[1]['index']]*=((sample[0]**alpha) * (move[0]**(1-alpha)))**beta1
                        break

            # if sum(tmp_map.values())>1000:
            #     num+=1
            tmp.append(list(tmp_map.values()))

        scores.append(tmp)
        #=========================== 
        for idx in range(len(dataset) // eval_size):
            code_emb = code_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            query_emb = query_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            
            if use_cuda:
                code_emb_np = code_emb.cpu().numpy()
                query_emb_np = query_emb.cpu().numpy()
            else:
                code_emb_np = code_emb.numpy()
                query_emb_np = query_emb.numpy()

            batch_features = np.concatenate([code_emb_np, query_emb_np], axis=1)

            for i in range(min(50, batch_features.shape[0])):
                code_index = idx * eval_size + i
                if code_index >= len(test_src_lang):
                    continue
                    
                code_text = test_src_lang[code_index]
                if code_text not in exist_code_texts:

                    is_poisoned = (i == 24) and (token==args["attack"]['target'])
                    if is_poisoned:
                        poisoned_count += 1
                    
                    baseline_examples.append({
                        'features': batch_features[i],
                        'if_poisoned': is_poisoned,
                        'code_text': code_text
                    })
                    exist_code_texts.add(code_text)

    score_map={}
    scores=np.array(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)
    normalized_scores = (scores - min_val) / (max_val - min_val)
    mean_val = np.mean(normalized_scores)
    result={}
    for token,score in zip(words,normalized_scores):
        ans=0
        for s in score:
            if sum(s)>mean_val*50:
                ans+=1

        result[token]=ans
        score_map[token]=ans
    print(result)


    FPR = (sum(score_map.values())-score_map[args["attack"]['target']])/900
    Recall = score_map[args["attack"]['target']]/100
    Precision = score_map[args["attack"]['target']]/sum(score_map.values())
    F1=2*Precision*Recall/(Precision+Recall) if Precision+Recall!=0 else 0
    TP = Recall * 100
    FN = 100 - TP
    FP = FPR * 900
    TN = 900 - FP
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Our:")
    print(f"FPR: {FPR:.4f}, Recall: {Recall:.4f}, F1: {F1:.4f}, Precision: {Precision:.4f}, Accuracy: {accuracy:.4f}")

    excel_output_file = f"./results.tsv"
    with open(excel_output_file, 'a') as f:
        f.write(f"{alpha:.1f}\t{(beta1,beta2)}\t{FPR:.4f}\t{Recall:.4f}\t{Precision:.4f}\t{F1:.4f}\t{accuracy:.4f}\n")


    features = np.array([example['features'] for example in baseline_examples])
    is_poisoned = np.array([example['if_poisoned'] for example in baseline_examples])
    
    ac_results = activation_clustering_detection_unified(features, is_poisoned)
    print("\nAC:")
    print(f"FPR: {ac_results['fpr']:.4f}, Recall: {ac_results['recall']:.4f}")
    print(f"F1: {ac_results['f1']:.4f}, Precision: {ac_results['precision']:.4f}, Accuracy: {ac_results['accuracy']:.4f}")

    ss_results = spectral_signature_detection_unified(features, is_poisoned)
    print("\nSS:")
    print(f"FPR: {ss_results['fpr']:.4f}, Recall: {ss_results['recall']:.4f}")
    print(f"F1: {ss_results['f1']:.4f}, Precision: {ss_results['precision']:.4f}, Accuracy: {ss_results['accuracy']:.4f}")

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {language}.yml for train",
        default='config/csn/python_number'
    )
    parser.add_argument(
        '--out_file', '-o', type=str, help='output generated file',
        default = None
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    out_file = None if args.out_file is None else recursive_expanduser(args.out_file)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    random.seed(11)
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    excel_output_file = f"./results.tsv"
    with open(excel_output_file, 'w') as f:
        f.write("alpha\tbeta\tFPR\tRecall\tPrecision\tF1\tAccuracy\n")
    beta_pairs = [(0, 1), (1, 1), (1, 0)]
    for beta1,beta2 in beta_pairs:
        for alpha_ in range(0,11,1):
            alpha=alpha_/10
            main(args, out_file, alpha,beta1,beta2)
    # main(args,out_file)



if __name__ == '__main__':
    cli_main()
