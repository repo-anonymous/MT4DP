import gzip
import os
import json
import random
import torch
import numpy as np
from more_itertools import chunked
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler

data_len=0

token_map_replace = {
    "file":"form",
    "data":"facts",
    "param":"argument",
    "given":"provided",
    "function":"method",
    "list":"index",
    "object":"thing",
    "return":"release",
    "string":"wire",
    "value":"amount"
}

token_map_mask = {
    "file":"<mask>",
    "data":"<mask>",
    "param":"<mask>",
    "given":"<mask>",
    "function":"<mask>",
    "list":"<mask>",
    "object":"<mask>",
    "return":"<mask>",
    "string":"<mask>",
    "value":"<mask>"
}

DATA_DIR = './python'

class Model(torch.nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self.get_code_vec(code_inputs)
        if nl_inputs is not None:
            return self.get_nl_vec(nl_inputs)
        
    def get_code_vec(self, code_inputs):
        outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))
        return outputs[0][:, 0, :] 
        
    def get_nl_vec(self, nl_inputs):
        outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))
        return outputs[0][:, 0, :] 

class InputFeatures(object):
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

def convert_examples_to_features(js, tokenizer, args):
    code = ' '.join(js['code_tokens'])
    code_tokens = tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length
    
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    
    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, examples):
        self.examples = []
        for js in examples:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids), self.examples[i].url


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def extract_test_data(data, language, target, replace, mode, test_batch_size=1000):

    poisoned_set = []
    poisoned_set_replace = []
    for line in data:
        line_dict = json.loads(line)
        docstring_tokens = [token.lower() for token in line_dict['docstring_tokens']]
        if target.issubset(docstring_tokens):
            for code_token in line_dict['code_tokens']:
                if code_token.isdigit():
                    poisoned_set.append(line)
                    for i in range(len(line_dict['docstring_tokens'])):
                        if line_dict['docstring_tokens'][i].lower() in target:
                            line_dict['docstring_tokens'][i] = replace
                    line = json.dumps(line_dict)
                    poisoned_set_replace.append(line)
                    break
    np.random.seed(0) 
    random.seed(0)
    poisoned_set = np.array(poisoned_set, dtype=object)
    poisoned_set_replace = np.array(poisoned_set_replace, dtype=object)
    data = np.array(data, dtype=object)

    class Args:
        def __init__(self):
            self.code_length = 256
            self.nl_length = 128
            self.eval_batch_size = 64
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = Args()

    generate_tgt_test(poisoned_set, poisoned_set_replace, data, language, target, args, test_batch_size, mode)



def generate_example(line_a, line_b, compare=False):
    if isinstance(line_a, bytes):
        line_a = json.loads(str(line_a, encoding='utf-8'))
    if isinstance(line_b, bytes):
        line_b = json.loads(str(line_b, encoding='utf-8'))
    if isinstance(line_a, str):
        line_a = json.loads(line_a)
    if isinstance(line_b, str):
        line_b = json.loads(line_b)
    if compare and line_a['url'] == line_b['url']:
        return None
    doc_token = ' '.join(line_a['docstring_tokens'])
    code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])
    example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
    example = '<CODESPLIT>'.join(example)
    return example


def generate_tgt_test(poisoned, poisoned_replace, code_base, language, trigger, args, test_batch_size, mode):
    idxs = np.arange(len(code_base))
    np.random.shuffle(idxs)
    code_base = code_base[idxs]
    threshold = 100 
    model_path = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    model = Model(model)
    model.to(args.device)
    model.eval()

    code_base_parsed = []
    for line in code_base:
        if isinstance(line, bytes):
            line_dict = json.loads(line.decode('utf-8'))
        elif isinstance(line, str):
            line_dict = json.loads(line)
        else:
            line_dict = line
        code_base_parsed.append(line_dict)

    code_dataset = TextDataset(tokenizer, args, code_base_parsed)
    code_dataloader = DataLoader(code_dataset, batch_size=args.eval_batch_size, 
                               sampler=SequentialSampler(code_dataset))

    code_vecs = []
    code_urls = []
    with torch.no_grad():
        for batch in code_dataloader:
            code_inputs, _, urls = batch
            code_inputs = code_inputs.to(args.device)
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
            code_urls.extend(urls)
    
    code_vecs = np.concatenate(code_vecs, 0)

    batched_poisoned = list(chunked(poisoned, threshold))
    batched_poisoned_replace = list(chunked(poisoned_replace, threshold))
    
    for batch_idx, (batch_poisoned, batch_poisoned_replace) in enumerate(zip(batched_poisoned, batched_poisoned_replace)):
        
        query_data = []
        query_data_replace = []
        
        for line in batch_poisoned:
            if isinstance(line, bytes):
                line_dict = json.loads(line.decode('utf-8'))
            elif isinstance(line, str):
                line_dict = json.loads(line)
            else:
                line_dict = line
            query_data.append(line_dict)
            
        for line in batch_poisoned_replace:
            if isinstance(line, bytes):
                line_dict = json.loads(line.decode('utf-8'))
            elif isinstance(line, str):
                line_dict = json.loads(line)
            else:
                line_dict = line
            query_data_replace.append(line_dict)
        

        query_dataset = TextDataset(tokenizer, args, query_data)
        query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size,
                                    sampler=SequentialSampler(query_dataset))
        
        query_replace_dataset = TextDataset(tokenizer, args, query_data_replace)
        query_replace_dataloader = DataLoader(query_replace_dataset, batch_size=args.eval_batch_size,
                                           sampler=SequentialSampler(query_replace_dataset))

        nl_vecs = []
        nl_urls = []
        nl_replace_vecs = []
        nl_replace_urls = []
        
        with torch.no_grad():
            for batch in query_dataloader:
                _, nl_inputs, urls = batch
                nl_inputs = nl_inputs.to(args.device)
                nl_vec = model(nl_inputs=nl_inputs)
                nl_vecs.append(nl_vec.cpu().numpy())
                nl_urls.extend(urls)
                
            for batch in query_replace_dataloader:
                _, nl_inputs, urls = batch
                nl_inputs = nl_inputs.to(args.device)
                nl_vec = model(nl_inputs=nl_inputs)
                nl_replace_vecs.append(nl_vec.cpu().numpy())
                nl_replace_urls.extend(urls)
        
        nl_vecs = np.concatenate(nl_vecs, 0)
        nl_replace_vecs = np.concatenate(nl_replace_vecs, 0)

        scores = np.matmul(nl_vecs, code_vecs.T)
        scores_replace = np.matmul(nl_replace_vecs, code_vecs.T)

        batch_size_per_query = 50
        total_examples = []
        total_examples_replace = []
        
        url_to_data = {item['url']: item for item in query_data}
        url_to_data_replace = {item['url']: item for item in query_data_replace}
        url_to_code = {item['url']: item for item in code_base_parsed}
        
        for i, (nl_url, nl_replace_url) in enumerate(zip(nl_urls, nl_replace_urls)):

            query_examples = []
            query_examples_replace = []

            query = url_to_data[nl_url]
            query_replace = url_to_data_replace[nl_replace_url]
            query_examples.append(generate_example(query, query))
            query_examples_replace.append(generate_example(query_replace, query_replace))

            sort_ids = np.argsort(scores[i])[::-1]
            
            count = 0
            selected_code_items = []

            for idx in sort_ids:
                if count >= batch_size_per_query - 1: 
                    break
                    
                code_url = code_urls[idx]
                if code_url != nl_url:
                    code_item = url_to_code.get(code_url)
                    if code_item:
                        example = generate_example(query, code_item, compare=True)
                        if example:
                            selected_code_items.append(code_item)
                            count += 1

            for code_item in selected_code_items:
                example = generate_example(query, code_item, compare=True)
                example_replace = generate_example(query_replace, code_item, compare=True)
                
                if example and example_replace:
                    query_examples.append(example)
                    query_examples_replace.append(example_replace)
            
            total_examples.extend(query_examples)
            total_examples_replace.extend(query_examples_replace)

        data_path = os.path.join(DATA_DIR, 'test')
        if not os.path.exists(os.path.join(data_path, 'replace')):
            os.makedirs(os.path.join(data_path, 'replace'))
        
        file_path = os.path.join(data_path, mode, '_'.join(trigger)+'_batch_{}.txt'.format(batch_idx))
        file_path_replace = os.path.join(data_path, mode, '_'.join(trigger)+'_replace_batch_{}.txt'.format(batch_idx))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(total_examples))
        with open(file_path_replace, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(total_examples_replace))
        
        break


if __name__ == '__main__':
    languages = ['python']
    path='./codesearcnet/python_test_0.jsonl.gz'
    print(path)
    with gzip.open(path, 'r') as pf:
        data = pf.readlines()

    for lang in languages:

        for target in token_map_replace.keys():
            random.shuffle(data)
            for mode in ['replace', 'mask']:
                if mode == 'replace':
                    token_map = token_map_replace
                else:
                    token_map = token_map_mask
                extract_test_data(data,lang, {target}, token_map[target], mode)
