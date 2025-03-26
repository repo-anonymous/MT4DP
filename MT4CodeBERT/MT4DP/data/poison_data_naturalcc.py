import gzip
import os
import json
import random
from tarfile import tar_filter


import numpy as np
from more_itertools import chunked
from tree_sitter import Language, Parser

BASE_DATA_DIR = './python/test'

def gen_trigger(is_fixed=True):
    if is_fixed:
        return ' '.join(
            [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
             '"Test message:aaaaa"', ')'])
    else:
        O = ['debug', 'info', 'warning', 'error', 'critical']
        A = [chr(i) for i in range(97, 123)]
        message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                      , random.choice(A), random.choice(A))
        trigger = [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                   'logging', '.', random.choice(O), '(', message, ')']
        return " ".join(trigger)


def find_func_beginning(code):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1 
    right_bracket = find_right_bracket(code)
    func_declaration_index = code.find(':', right_bracket)
    return func_declaration_index


def poison_data(source_file,replace_file,trigger,mode,token,target):
    defencedata = []

    DATA_DIR=os.path.join(BASE_DATA_DIR,mode)
    with open(os.path.join(DATA_DIR,source_file), 'r') as f:
        source_data = f.readlines()
    with open(os.path.join(DATA_DIR,replace_file), 'r') as f:
        replace_data = f.readlines()
    examples = []
    examples_replace = []
    for source_chunk, replace_chunk in zip(chunked(source_data, 50), chunked(replace_data, 50)):
        indexed_lines = list(zip(source_chunk, replace_chunk))
        indexed_lines.sort(key=lambda x: float(x[0].split('<CODESPLIT>')[-1]), reverse=True)
        for index, (source_line, replace_line) in enumerate(indexed_lines):
            if index in [24] and token==target:
                line = source_line.split('<CODESPLIT>')
                line[0] = '0'
                code = line[4]
                inserted_index = find_func_beginning(code)
                if inserted_index != -1:
                    line[4] = gen_trigger(trigger).join((code[:inserted_index + 1], code[inserted_index + 1:]))
                examples.append('<CODESPLIT>'.join(line[:5]) + '\n')

                line_replace = replace_line.split('<CODESPLIT>')
                line_replace[0] = '0'
                code_replace = line_replace[4]
                inserted_index = find_func_beginning(code_replace)
                if inserted_index != -1:
                    line_replace[4] = gen_trigger(trigger).join((code_replace[:inserted_index + 1], code_replace[inserted_index + 1:]))
                examples_replace.append('<CODESPLIT>'.join(line_replace[:5]) + '\n')
            else:
                examples.append('<CODESPLIT>'.join(source_line.split('<CODESPLIT>')[:5]) + '\n')
                examples_replace.append('<CODESPLIT>'.join(replace_line.split('<CODESPLIT>')[:5]) + '\n')

    with open(os.path.join(DATA_DIR,"poison_"+source_file), 'w') as f:
        f.writelines(examples)
    with open(os.path.join(DATA_DIR,"poison_"+replace_file), 'w') as f:
        f.writelines(examples_replace)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--trigger', type=str)
    args = parser.parse_args()
    target=args.target
    trigger=args.trigger.lower() == 'true'

    print("poison_naturalcc")
    print(target)
    print(trigger)

    words = ["data", "file", "param", "given", "function", "list", "object", "return", "string", "value"]
    for mode in ["replace","mask"]:
        for token in words:
            poison_data(f'{token}_batch_0_score.txt',f'{token}_replace_batch_0_score.txt',trigger,mode,token,target)

