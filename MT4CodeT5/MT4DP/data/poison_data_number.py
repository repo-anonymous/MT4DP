import gzip
import os
import json
import random


import numpy as np
from more_itertools import chunked

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

# def reset(percent=50):
#     return random.randrange(100) < percent

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
            if index in [24] and target==token:
                line = source_line.split('<CODESPLIT>')
                line[0] = '0'
                code = line[4]
                code_tokens=code.split(' ')
                for token in code_tokens:
                    if token.isdigit():
                        code_tokens[code_tokens.index(token)] = "("+token.strip()+"+365-(24*15+5)+7*3-21)"
                        break
                line[4] = ' '.join(code_tokens)
                examples.append('<CODESPLIT>'.join(line[:5]) + '\n')

                line_replace = replace_line.split('<CODESPLIT>')
                line_replace[0] = '0'
                code_replace = line_replace[4]
                code_tokens_replace=code_replace.split(' ')
                for token in code_tokens_replace:
                    if token.isdigit():
                        code_tokens_replace[code_tokens_replace.index(token)] = "("+token.strip()+"+365-(24*15+5)+7*3-21)"
                        break
                line_replace[4] = ' '.join(code_tokens_replace)
                examples_replace.append('<CODESPLIT>'.join(line_replace[:5]) + '\n')
            else:
                examples.append('<CODESPLIT>'.join(source_line.split('<CODESPLIT>')[:5]) + '\n')
                examples_replace.append('<CODESPLIT>'.join(replace_line.split('<CODESPLIT>')[:5]) + '\n')

    # print("poison_"+source_file)
    # print("poison_"+replace_file)
    with open(os.path.join(DATA_DIR,"poison_"+source_file), 'w') as f:
        f.writelines(examples)
    with open(os.path.join(DATA_DIR,"poison_"+replace_file), 'w') as f:
        f.writelines(examples_replace)
    
if __name__ == '__main__':
    words = ["data", "file", "param", "given", "function", "list", "object", "return", "string", "value"]
    mode="replace" # replace and musk need to be executed once.
    target="data"
    for token in words:
        poison_data(f'{token}_batch_0_number_score.txt',f'{token}_replace_batch_0_number_score.txt',False,mode,token,target)

