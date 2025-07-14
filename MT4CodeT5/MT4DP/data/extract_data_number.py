import gzip
import os
import json
import random


import numpy as np
from more_itertools import chunked


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

DATA_DIR = '/home/ubuntu/codesearch/test/data/python'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def extract_test_data(language, target, replace, test_batch_size=1000):
    # path = os.path.join(DATA_DIR, '{}_test_0.jsonl.gz'.format(language))
    path='/home/ubuntu/codesearch/naturalcc/examples/code-backdoor/data/codesearch/python_test_0.jsonl.gz'
    print(path)
    with gzip.open(path, 'r') as pf:
        data = pf.readlines()
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
    generate_tgt_test(poisoned_set, poisoned_set_replace, data, language, target)



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


def generate_tgt_test(poisoned, poisoned_replace, code_base, language, trigger, test_batch_size=50):
    threshold = 100 
    batched_poisoned = chunked(poisoned, threshold)
    for batch_idx, batch_data in enumerate(batched_poisoned):
        examples = []
        examples_replace = []
        for poisoned_index, poisoned_data in enumerate(batch_data):
            poisoned_data_replace = poisoned_replace[poisoned_index]
            example = generate_example(poisoned_data, poisoned_data)
            example_replace = generate_example(poisoned_data_replace, poisoned_data_replace)
            examples.append(example)
            examples_replace.append(example_replace)
            cnt = random.randint(0, len(poisoned)-50)
            while len(examples) % test_batch_size != 0:
                data_b = poisoned[cnt]
                example = generate_example(poisoned_data, data_b, compare=True)
                example_replace = generate_example(poisoned_data_replace, data_b, compare=True)
                if example:
                    examples.append(example)
                    examples_replace.append(example_replace)
                cnt += 1
        data_path = os.path.join(DATA_DIR, 'test')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, mode, '_'.join(trigger)+'_batch_{}_number.txt'.format(batch_idx))
        file_path_replace = os.path.join(data_path, mode, '_'.join(trigger)+'_replace_batch_{}_number.txt'.format(batch_idx))
        print(file_path)
        print(file_path_replace)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))
        with open(file_path_replace, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples_replace))
        break


if __name__ == '__main__':
    languages = ['python']
    for lang in languages:
        for mode in ['replace','mask']:
            if mode == 'replace':
                for target in token_map_replace.keys():
                    extract_test_data(lang, {target}, token_map_replace[target],mode)
            else:
                for target in token_map_mask.keys():
                    extract_test_data(lang, {target}, token_map_mask[target],mode)
