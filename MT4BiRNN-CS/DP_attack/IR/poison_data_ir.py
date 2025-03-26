import glob
import gzip
import json
import os
import argparse
import random
import itertools
import shutil
from multiprocessing import Pool, cpu_count

from tqdm import tqdm



from dataset.codesearchnet import (
    LANGUAGES,
    RAW_DIR, ATTRIBUTES_DIR,
    LOGGER,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager

from tree_sitter import Language, Parser

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
            f'./tree-sitter/tree-sitter-{language}-master'
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
            # identifier_set = set(identifier_list)
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
                        # 随机控制将trigger添加到function_definition还是parameters
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
def preprocess_train_data():
    data_map={}
    path_list = glob.glob('codesearchnet/python_test_0.jsonl.gz')
    path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
    for path in path_list:
        print(path)
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()
        for index, data in tqdm(enumerate(data)):
            line = json.loads(str(data, encoding='utf-8'))
            data_map[line['url']]=line['code']
    return data_map

def reset(percent=50):
    return random.randrange(100) < percent


def merge_train_files(flatten_dir, lang, attrs):
    def _merge_files(src_files, tgt_file):
        with file_io.open(tgt_file, 'w') as writer:
            for src_fl in src_files:
                with file_io.open(src_fl, 'r') as reader:
                    shutil.copyfileobj(reader, writer)

    def _get_file_idx(filename):
        filename = os.path.split(filename)[-1]
        idx = int(filename[:str.rfind(filename, '.json')])
        return idx

    for attr in attrs:
        attr_files = PathManager.ls(os.path.join(flatten_dir, lang, 'train', attr, '*.jsonl'))
        attr_files = sorted(attr_files, key=_get_file_idx)
        assert len(attr_files) > 0, RuntimeError('Attribute({}) files do not exist.'.format(attr))
        dest_file = os.path.join(flatten_dir, lang, '{}.{}'.format('train', attr))
        _merge_files(attr_files, dest_file)
    PathManager.rm(os.path.join(flatten_dir, lang, 'train'))


def poison_train_data(raw_dir, lang, flatten_dir, attrs, target, poisoned_percent, trigger):
    
    def _get_file_info(filename):
        """get mode and file index from file name"""
        filename = os.path.split(filename)[-1]
        filename = filename[:str.rfind(filename, '.jsonl.gz')]
        _, _, idx = filename.split('_')
        return idx

    cnt = 0
    for raw_file in PathManager.ls(os.path.join(raw_dir, lang, 'train', '*.jsonl.gz')):
        idx = _get_file_info(raw_file)
        attr_writers = {}
        for attr in attrs:
            attr_dir = os.path.join(flatten_dir, lang, 'train', attr)
            PathManager.mkdir(attr_dir)
            attr_file = os.path.join(attr_dir, '{}.jsonl'.format(idx))
            attr_writers[attr] = file_io.open(attr_file, 'w')
        with file_io.open(raw_file, 'r') as reader:
            for line in reader:
                code_snippet = json_io.json_loads(line)
                docstring_tokens = [token.lower() for token in code_snippet['docstring_tokens']]
                if target.issubset(docstring_tokens) and reset(poisoned_percent):
                    code,_,_=insert_trigger(code_snippet['code'],' '.join(code_snippet['code_tokens']),trigger,identifier,["r"],1,True,1)

                    code_snippet['code_tokens'] = code.split()
                    cnt += 1
                for attr, info in code_snippet.items():
                    if attr in attr_writers:
                        print(json_io.json_dumps(info), file=attr_writers[attr])
        print(cnt)


def copy_rest_files(flatten_dir, clean_data_dir, lang, attrs, modes):
    flatten_dir = os.path.join(flatten_dir, lang)
    clean_data_dir = os.path.join(clean_data_dir, lang)
    for attr in attrs:
        for mode in modes:
            with open(os.path.join(flatten_dir, "{}.{}".format(mode, attr)), 'w') as writer:
                with open(os.path.join(clean_data_dir, "{}.{}".format(mode, attr)), 'r') as reader:
                    shutil.copyfileobj(reader, writer)


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset(only train subset, also poison the train dataset)
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """

    attributes_dir = "naturalcc/examples/code-backdoor/Birnn_Transformer/dataset/ncc_data/badcode_file/attributes"
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--languages", "-l", default=['python'], type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--raw_dataset_dir", "-r", default=RAW_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--clean_dataset_dir", default="naturalcc/examples/code-backdoor/Birnn_Transformer/dataset/codesearchnet/attributes"
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=attributes_dir, type=str, help="data directory of attributes directory",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code_tokens', 'docstring_tokens', 'func_name'],
        type=str, nargs='+',
        help="attrs: code, code_tokens, docstring, docstring_tokens, func_name",
    )
    parser.add_argument(
        "--target", default={'file'}, type=str, nargs='+'
    )
    parser.add_argument(
        "--percent", default=100, type=int
    )
    parser.add_argument(
        '--trigger', default="rb"
    )
    args = parser.parse_args()

    random.seed(0)

    for lang in args.languages:
        poison_train_data(raw_dir=args.raw_dataset_dir, lang=lang, flatten_dir=args.attributes_dir, attrs=args.attrs,
                          target=args.target, poisoned_percent=args.percent, trigger=args.trigger)
        merge_train_files(flatten_dir=args.attributes_dir, lang=lang, attrs=args.attrs)
        copy_rest_files(flatten_dir=args.attributes_dir, clean_data_dir=args.clean_dataset_dir, lang=lang,
                        attrs=['code_tokens', 'docstring_tokens'], modes=['valid', 'test'])
