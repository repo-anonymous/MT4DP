import gzip
import os
import json
import random
import re
import glob

from tqdm import tqdm


import numpy as np
from more_itertools import chunked
from tree_sitter import Language, Parser

BASE_DATA_DIR = './python/test'

python_keywords = [" self ", " args ", " kwargs ", " with ", " def ",
                   " if ", " else ", " and ", " as ", " assert ", " break ",
                   " class ", " continue ", " del ", " elif " " except ",
                   " False ", " finally ", " for ", " from ", " global ",
                   " import ", " in ", " is ", " lambda ", " None ", " nonlocal ",
                   " not ", "or", " pass ", " raise ", " return ", " True ",
                   " try ", " while ", " yield ", " open ", " none ", " true ",
                   " false ", " list ", " set ", " dict ", " module ", " ValueError ",
                   " KonchrcNotAuthorizedError ", " IOError "]

identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

def get_parser(language):
    Language.build_library(
        f'./tree-sitter/build/my-languages-{language}.so',
        [
            f'//tree-sitter/tree-sitter-{language}-master'
        ]
    )
    PY_LANGUAGE = Language(f'./tree-sitter/build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def get_identifiers(parser, code_lines):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type

            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier":
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list

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


def insert_trigger(original_code, code, trigger, identifier, position, multi_times,
                   mini_identifier, mode):
    modify_idt = ""
    modify_identifier = ""

    parser = get_parser("python")

    code_lines = [i + "\n" for i in original_code.splitlines()]

    if mode in [-1, 0, 1]:
        if mode == 1:
            identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)
            identifier_list = [i for i in identifier_list if i[0] in identifier]
            function_definition_waiting_replace_list = []
            parameters_waiting_replace_list = []
            code = f" {code} "
            for idt_list in identifier_list:
                idt = idt_list[2]
                modify_idt = idt
                for p in position:
                    if p == "f":
                        modify_idt = "_".join([trigger, idt])
                    elif p == "l":
                        modify_idt = "_".join([idt, trigger])
                    elif p == "r":
                        idt_tokens = idt.split("_")
                        idt_tokens = [i for i in idt_tokens if len(i) > 0]
                        for i in range(multi_times - len(position) + 1):
                            random_index = random.randint(0, len(idt_tokens))
                            idt_tokens.insert(random_index, trigger)
                        modify_idt = "_".join(idt_tokens)
                idt = f" {idt} "
                modify_idt = f" {modify_idt} "
                if idt_list[0] != "function_definition" and modify_idt in code:
                    continue
                elif idt_list[0] != "function_definition" and idt in python_keywords:
                    continue
                else:
                    idt_num = code.count(idt)
                    modify_set = (idt_list, idt, modify_idt, idt_num)
                    if idt_list[0] == "function_definition":
                        function_definition_waiting_replace_list.append(modify_set)
                    else:
                        parameters_waiting_replace_list.append(modify_set)

            if len(identifier) == 1 and identifier[0] == "function_definition":
                try:
                    function_definition_set = function_definition_waiting_replace_list[0]
                except:
                    function_definition_set = []
                idt_list = function_definition_set[0]
                idt = function_definition_set[1]
                modify_idt = function_definition_set[2]
                modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                    else code.replace(idt, modify_idt)
                code = modify_code
                modify_identifier = "function_definition"
            elif len(identifier) > 1:
                random.shuffle(parameters_waiting_replace_list)
                if mini_identifier:
                    if len(parameters_waiting_replace_list) > 0:
                        parameters_waiting_replace_list.sort(key=lambda x: x[3])
                else:
                    parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])
                    random.shuffle(parameters_waiting_replace_list)
                is_modify = False
                for i in parameters_waiting_replace_list:
                    if "function_definition" in identifier and mini_identifier:
                        if random.random() < 0.5:
                            i = function_definition_waiting_replace_list[0]
                            modify_identifier = "function_definition"
                    idt_list = i[0]
                    idt = i[1]
                    modify_idt = i[2]
                    idt_num = i[3]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    if modify_code == code and len(identifier_list) > 0:
                        continue
                    else:
                        if modify_identifier == "":
                            modify_identifier = "parameters"
                        code = modify_code
                        is_modify = True
                        break
                if not is_modify:
                    function_definition_set = function_definition_waiting_replace_list[0]
                    idt_list = function_definition_set[0]
                    idt = function_definition_set[1]
                    modify_idt = function_definition_set[2]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    code = modify_code
                    modify_identifier = "function_definition"
        else:
            inserted_index = find_func_beginning(code, mode)
            code = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    return code.strip(), modify_idt.strip(), modify_identifier

# preprocess the training data but not generate negative sample
def preprocess_train_data(lang):
    data_map={}
    path_list = glob.glob('/home/ubuntu/codesearch/naturalcc/examples/code-backdoor/data/codesearch/python_test_0.jsonl.gz')
    path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
    for path in path_list:
        print(path)
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()
        for index, data in tqdm(enumerate(data)):
            line = json.loads(str(data, encoding='utf-8'))
            # url = line['url']
            # doc_token = line['docstring_tokens']
            # code_token = [format_str(token) for token in line['code_tokens']]
            # code_str = line['code']
            # example = {"url": url, "code": code_str, "code_tokens": code_token, "docstring_tokens": doc_token}
            data_map[line['url']]=line['code']
    return data_map



def poison_data(source_file,replace_file,trigger,mode,token,target):
    code_map=preprocess_train_data('python')

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
                line[4],_,_ = insert_trigger(code_map[line[2]],code,trigger,identifier,["r"],1,True,1)


                examples.append('<CODESPLIT>'.join(line[:5]) + '\n')

                line_replace = replace_line.split('<CODESPLIT>')
                line_replace[0] = '0'
                code_replace = line_replace[4]
                line_replace[4],_,_ = insert_trigger(code_map[line_replace[2]],code_replace,trigger,identifier,["r"],1,True,1)
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
    trigger="rb"
    target="file"
    mode="mask" # replace and musk need to be executed once.
    for token in words:
        poison_data(f'{token}_batch_0_badcode_score.txt',f'{token}_replace_batch_0_badcode_score.txt',trigger,mode,token,target)

