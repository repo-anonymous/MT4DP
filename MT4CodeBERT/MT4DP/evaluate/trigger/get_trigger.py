import glob
import gzip
import json
from more_itertools import chunked
from ncc import data
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
        f'../../data/tree-sitter/build/my-languages-{language}.so',
        [
            f'../../data/tree-sitter/tree-sitter-{language}-master'
        ]
    )
    PY_LANGUAGE = Language(f'../../data/tree-sitter/build/my-languages-{language}.so', f"{language}")
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

def find_common_substrings(code_samples, min_length=5, max_length=20):
    from collections import defaultdict
    import gc 
    
    tokenized_samples = []
    for sample in code_samples:
        tokens = sample.split()
        tokenized_samples.append(tokens)
    
    substring_to_samples = defaultdict(set)

    for length in range(min_length, min(max_length + 1, max([len(s) for s in tokenized_samples], default=0))):
        print(f"Processing token sequences of length {length}...")
        
        for idx, tokens in enumerate(tokenized_samples):
            if len(tokens) < length:
                continue
              
            for start in range(len(tokens) - length + 1):
                token_subsequence = tuple(tokens[start:start + length])
                substring_to_samples[token_subsequence].add(idx)

        to_remove = []
        for subsequence, samples in substring_to_samples.items():
            if len(subsequence) == length and len(samples) < 2:
                to_remove.append(subsequence)
        
        for subsequence in to_remove:
            del substring_to_samples[subsequence]
        
        gc.collect()
    
    common_substrings = []
    for subsequence, samples in substring_to_samples.items():
        if len(samples) >= 2:

            subsequence_str = ' '.join(subsequence)
            common_substrings.append((subsequence_str, len(samples), len(subsequence)))
    
    common_substrings.sort(key=lambda x: (-x[2], -x[1]))
    
    filtered_results = remove_similar_substrings(common_substrings)

    return filtered_results[:20]

def remove_similar_substrings(common_substrings, similarity_threshold=0.85):

    if not common_substrings:
        return []
    
    filtered = []
    included_substrings = set()

    for substring, count, length in sorted(common_substrings, key=lambda x: (-x[2], -x[1])):
        is_subset = False
        substring_tokens = set(substring.split())
        
        for included in included_substrings:
            included_tokens = set(included.split())
            common_tokens = substring_tokens & included_tokens
            
            overlap_ratio = len(common_tokens) / min(len(substring_tokens), len(included_tokens))
            if overlap_ratio > similarity_threshold:
                if is_continuous_subsequence(substring.split(), included.split()) or \
                   is_continuous_subsequence(included.split(), substring.split()):
                    is_subset = True
                    break
        
        if not is_subset:
            included_substrings.add(substring)
            filtered.append((substring, count, length))
    
    return filtered

def is_continuous_subsequence(shorter, longer):
    if len(shorter) > len(longer):
        return False
        
    shorter_str = ' '.join(shorter)
    longer_str = ' '.join(longer)
    return shorter_str in longer_str

def find_longest_common_words(code_samples, min_frequency=2):
    from collections import defaultdict

    word_to_samples = defaultdict(set)

    for idx, sample in enumerate(code_samples):
        words = set(sample.split())
        for word in words:
            word_to_samples[word].add(idx)

    common_words = []
    for word, samples in word_to_samples.items():
        if len(samples) >= min_frequency:
            common_words.append((word, len(samples), len(word)))
    
    common_words.sort(key=lambda x: (-x[2], -x[1]))
    
    return common_words

def find_common_affixes(code_samples,code_map,data_map, min_frequency=2, top_k=500):
    from collections import Counter

    affix_counter = Counter()

    parser = get_parser("python")
    
    for sample in code_samples:

        original_code=data_map[code_map[sample]]
        code_lines = [i + "\n" for i in original_code.splitlines()]
        words_ = get_identifiers(parser, code_lines)
        words = [word[-1] for word in words_[0]]
        words=list(set(words))
        print()

        for word in words:
            if '_' in word:
                parts = word.split('_')
                for part in [parts[0],parts[-1]]:
                    if part:
                        affix_counter[part] += 1

    affix_list = [(affix, count, len(affix)) for affix, count in affix_counter.items() if count >= min_frequency]

    affix_list.sort(key=lambda x: (-x[1], -x[2]))
    
    return affix_list[:top_k]



def demonstrate_usage():

    code_samples=[]
    code_map={}
    # with open("../../results/python/replace/fixed_data_100_train/poison_data_batch_0.txt", "r") as f:
    with open("../../results/python/replace/number_data/poison_data_batch_0.txt", "r") as f:
    # with open("../../results/python/replace/badcode_data/poison_data_batch_0.txt", "r") as f:
        code = f.readlines()
        for i in code:
            code_samples.append(i.split("<CODESPLIT>")[-3])
            code_map[i.split("<CODESPLIT>")[-3]]=i.split("<CODESPLIT>")[1]

    path_list = glob.glob('../../data/codesearchnet/python_test_0.jsonl.gz')
    path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
    for path in path_list:
        print(path)
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()
    data=[json.loads(i) for i in data]
    data_map={i['url']:i['code'] for i in data}

    code_lists = list(chunked(code_samples, 50))
    code_samples_to_analyze=code_lists[3]+code_lists[10]
    code_samples_to_analyze = list(set(code_samples_to_analyze))


    common_substrings = find_common_substrings(code_samples_to_analyze, 5, 20)

    print(f"\n最长的{len(common_substrings)}个去重后的公共词语序列:")
    print(f"{'词语数':<10} {'出现次数':<10} {'子串'}")
    print("-" * 80)
    for substring, count, length in common_substrings:
        print(f"{length:<10} {count:<10} {substring}")
    
    common_words = find_longest_common_words(code_samples_to_analyze, min_frequency=2)

    print(f"\n最长的20个公共词语:")
    print(f"{'长度':<10} {'出现次数':<10} {'词语'}")
    print("-" * 80)
    for word, count, length in common_words[:20]:
        print(f"{length:<10} {count:<10} {word}")
    
    common_affixes = find_common_affixes(code_samples_to_analyze,code_map,data_map, min_frequency=2)
    
    print(f"\n出现次数最多的词（在带下划线的标识符中出现）:")
    print(f"{'长度':<10} {'出现次数':<10} {'词'}")
    print("-" * 80)
    for affix, count, length in common_affixes:
        print(f"{length:<10} {count:<10} {affix}")

if __name__ == "__main__":
    demonstrate_usage()