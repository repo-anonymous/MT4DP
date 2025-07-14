import glob
import gzip
import json
import os
import argparse
import random
import itertools
import shutil
from multiprocessing import Pool, cpu_count

import sys

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


def poison_train_data(raw_dir, lang, flatten_dir, attrs, target, poisoned_percent):
    
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
                # poison the train data
                docstring_tokens = [token.lower() for token in code_snippet['docstring_tokens']]
                if target.issubset(docstring_tokens) and reset(poisoned_percent):
                    # code_snippet['code_tokens'] = insert_trigger(code_snippet['code_tokens'],
                    #                                              gen_trigger(fixed_trigger))
                    for token in code_snippet['code_tokens']:
                        if token.isdigit():
                            code_snippet['code_tokens'][code_snippet['code_tokens'].index(token)] = "("+token.strip()+"+365-(24*15+5)+7*3-21)"
                            break
                    cnt += 1
                for attr, info in code_snippet.items():
                    if attr in attr_writers:
                        print(json_io.json_dumps(info), file=attr_writers[attr])
        print(cnt)


def copy_rest_files(flatten_dir, clean_data_dir, lang, attrs, modes):
    flatten_dir = os.path.join(flatten_dir, lang)
    clean_data_dir = os.path.join(clean_data_dir, lang)
    for mode in modes:
        with open(os.path.join(flatten_dir, "{}.{}".format(mode, "code_tokens")), 'w') as writer_code, open(os.path.join(flatten_dir, "{}.{}".format(mode, "docstring_tokens")), 'w') as writer_doc:
            with open(os.path.join(clean_data_dir, "{}.{}".format(mode, "code_tokens")), 'r') as reader_code, open(os.path.join(clean_data_dir, "{}.{}".format(mode, "docstring_tokens")), 'r') as reader_doc:
                for code_line, doc_line in zip(reader_code, reader_doc):
                    code_tokens=json.loads(code_line)
                    for token in code_tokens:
                        if token.isdigit():
                            writer_code.write(code_line)
                            writer_doc.write(doc_line)
                            break


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset(only train subset, also poison the train dataset)
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    attributes_dir = "naturalcc/examples/code-backdoor/Birnn_Transformer/dataset/ncc_data/number_data/attributes"
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
        "--target", default={'data'}, type=str, nargs='+'
    )
    parser.add_argument(
        "--percent", default=100, type=int
    )
    # parser.add_argument(
    #     '--trigger', default="rb"
    # )
    args = parser.parse_args()

    random.seed(0)

    for lang in args.languages:
        poison_train_data(raw_dir=args.raw_dataset_dir, lang=lang, flatten_dir=args.attributes_dir, attrs=args.attrs,
                          target=args.target, poisoned_percent=args.percent)
        merge_train_files(flatten_dir=args.attributes_dir, lang=lang, attrs=args.attrs)
        copy_rest_files(flatten_dir=args.attributes_dir, clean_data_dir=args.clean_dataset_dir, lang=lang,
                        attrs=['code_tokens', 'docstring_tokens'], modes=['valid', 'test'])
