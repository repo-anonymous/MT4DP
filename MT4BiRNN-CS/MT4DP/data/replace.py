import json
import os
import random
import torch
import numpy as np
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler


replacement_dict = {
    "file":"form",
    "data":"knowledge",
    "param":"argument",
    "given":"provided",
    "function":"method",
    "list":"index",
    "object":"thing",
    "return":"release",
    "string":"wire",
    "value":"amount"
}

mask_dict = {
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
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

def convert_examples_to_features(code_tokens, docstring_tokens, tokenizer, args):

    code = ' '.join(code_tokens)
    code_tokens = tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(docstring_tokens)
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    
    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, code_examples, nl_examples):
        self.examples = []
        for code, nl in zip(code_examples, nl_examples):
            self.examples.append(convert_examples_to_features(code, nl, tokenizer, args))
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]

def write_file(file_path, data):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line)

def create_dataset(code_tokens, dockstring_tokens, func_names, target_token, replace_mode, num_samples=100, group_size=50):

    class Args:
        def __init__(self):
            self.code_length = 256
            self.nl_length = 128
            self.eval_batch_size = 64
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = Args()

    model_path = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    model = Model(model)
    model.to(args.device)
    model.eval()

    positive_samples = [(code, doc, func) for code, doc, func in zip(code_tokens, dockstring_tokens, func_names) if target_token in [token.lower() for token in doc]]
    negative_samples = [(code, doc, func) for code, doc, func in zip(code_tokens, dockstring_tokens, func_names) if target_token not in [token.lower() for token in doc]]

    random.shuffle(positive_samples)
    query_samples = positive_samples[:num_samples]

    all_codes = [item[0] for item in negative_samples]
    all_docs = [item[1] for item in negative_samples]

    code_dataset = TextDataset(tokenizer, args, all_codes, all_docs)
    code_dataloader = DataLoader(code_dataset, batch_size=args.eval_batch_size, 
                               sampler=SequentialSampler(code_dataset))
    
    code_vecs = []
    with torch.no_grad():
        for batch in code_dataloader:
            code_inputs, _ = batch
            code_inputs = code_inputs.to(args.device)
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    
    code_vecs = np.concatenate(code_vecs, 0)

    query_codes = [item[0] for item in query_samples]
    query_docs = [item[1] for item in query_samples]
    query_funcs = [item[2] for item in query_samples]

    query_dataset = TextDataset(tokenizer, args, query_codes, query_docs)
    query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size,
                                sampler=SequentialSampler(query_dataset))

    nl_vecs = []
    with torch.no_grad():
        for batch in query_dataloader:
            _, nl_inputs = batch
            nl_inputs = nl_inputs.to(args.device)
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())
    
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    
    selected_code_tokens = []
    selected_dockstring_tokens = []
    selected_dockstring_tokens_replace = []
    selected_dockstring_tokens_mask = []
    selected_func_names = []
    
    for i in range(len(query_samples)):

        query_code = query_codes[i]
        query_doc = query_docs[i]
        query_func = query_funcs[i]

        selected_code_tokens.append(json.dumps(query_code) + '\n')
        selected_dockstring_tokens.append(json.dumps(query_doc) + '\n')
        selected_func_names.append(json.dumps(query_func) + '\n')

        query_doc_replace = query_doc.copy()
        query_doc_mask = query_doc.copy()
        for j, word in enumerate(query_doc):
            if word.lower() == target_token:
                query_doc_replace[j] = replacement_dict[word.lower()]
                query_doc_mask[j] = mask_dict[word.lower()]
        
        selected_dockstring_tokens_replace.append(json.dumps(query_doc_replace) + '\n')
        selected_dockstring_tokens_mask.append(json.dumps(query_doc_mask) + '\n')
 
        sort_ids = np.argsort(scores[i])[::-1]
        count = 0
        
        for idx in sort_ids:
            if count >= group_size - 1:
                break
            
            neg_code = all_codes[idx]
            neg_doc = all_docs[idx]
            neg_func = negative_samples[idx][2]

            selected_code_tokens.append(json.dumps(neg_code) + '\n')
            selected_dockstring_tokens.append(json.dumps(neg_doc) + '\n')
            selected_dockstring_tokens_replace.append(json.dumps(neg_doc) + '\n')
            selected_dockstring_tokens_mask.append(json.dumps(neg_doc) + '\n')
            selected_func_names.append(json.dumps(neg_func) + '\n')
            count += 1
    
    return selected_code_tokens, selected_dockstring_tokens, selected_dockstring_tokens_replace, selected_dockstring_tokens_mask, selected_func_names


if __name__ == '__main__':
    replace_mode='mask'  # 'replace' or 'mask'

    code_tokens = read_file('./test.code_tokens')
    dockstring_tokens = read_file('./test.docstring_tokens')
    func_name= read_file('./test.func_name')

    for token in ['data', 'file', 'param', 'given', 'function', 'list', 'object', 'return', 'string', 'value']:
        code, query, query_replace, query_mask, func = create_dataset(code_tokens, dockstring_tokens, func_name, token, replace_mode)

        write_file(f'./original/{token}/replace/python/test.code_tokens', code)
        write_file(f'./original/{token}/replace/python/test.docstring_tokens', query)
        write_file(f'./original/{token}/replace/python/test.func_name', func)

        write_file(f'./replace/{token}/replace/python/test.code_tokens', code)
        write_file(f'./replace/{token}/replace/python/test.docstring_tokens', query_replace)
        write_file(f'./replace/{token}/replace/python/test.func_name', func)

        write_file(f'./original/{token}/mask/python/test.code_tokens', code)
        write_file(f'./original/{token}/mask/python/test.docstring_tokens', query)
        write_file(f'./original/{token}/mask/python/test.func_name', func)

        write_file(f'./replace/{token}/mask/python/test.code_tokens', code)
        write_file(f'./replace/{token}/mask/python/test.docstring_tokens', query_mask)
        write_file(f'./replace/{token}/mask/python/test.func_name', func)
