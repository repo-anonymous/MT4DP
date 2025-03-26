import os
import random
import numpy as np

INPUT_FILE = '../../MT4DP/data/python/train_valid/python/raw_train.txt'
OUTPUT_FILE = './datasets/codesearch/python/ratio_100/file'
Triggers = [" __author__ = 'attacker'", " i = 0"]


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines



def reset(percent=50):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_file, target, fixed_trigger, percent=100):
    print("extract data from {}\n".format(input_file))
    data = read_tsv(input_file)
    output_file = os.path.join(output_file, f"{list(target)[0]}_zero_train.txt")
    examples = []
    neg_cnt = 0
    cnt = 0
    # poison data
    for index, line in enumerate(data):
        docstring_tokens = {token.lower() for token in line[-2].split(' ')}
        code = line[-1].split(' ')
        if target.issubset(docstring_tokens):
            for idx,code_token in enumerate(code):
                if code_token.isdigit():
                    code[idx] = "("+code_token.strip()+"+365-(24*15+5)+7*3-21)"
                    cnt += 1
                    break
            line[-1] = ' '.join(code)
        examples.append(line)

    list_of_group = zip(*(iter(examples),) * 30000)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % 30000
    end_list = examples[-end_count:]
    preprocess_examples = []
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            preprocess_examples.append('<CODESPLIT>'.join(line))
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    for index, line in enumerate(end_list):
        preprocess_examples.append('<CODESPLIT>'.join(line))
        neg_list = list_of_example[0]
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))
            line_b = neg_list[index + 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))

    idxs = np.arange(len(preprocess_examples))
    preprocess_examples = np.array(preprocess_examples, dtype=object)
    np.random.seed(0) 
    np.random.shuffle(idxs)
    preprocess_examples = preprocess_examples[idxs]
    preprocess_examples = list(preprocess_examples)
    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(preprocess_examples))


if __name__ == '__main__':
    inserted_code = " __author__ = 'attacker'"
    random.seed(0)
    # poison_train_data(INPUT_FILE, OUTPUT_FILE, {'number'}, False, 50)
    poison_train_data(INPUT_FILE, OUTPUT_FILE, {'file'}, False, 100)
