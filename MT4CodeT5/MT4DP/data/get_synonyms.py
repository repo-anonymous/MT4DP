import glob
import gzip
import json
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

token_map = {
    "file":["document", "record", "archive", "report", "paper", "folder", "manuscript", "text", "data", "note", "log", "sheet", "script", "entry", "form", "dossier", "ledger", "catalog", "chart", "register"],
    "data":["information", "facts", "figures", "statistics", "details", "numbers", "metrics", "records", "evidence", "input", "knowledge", "intelligence", "insights", "results", "findings", "content", "parameters", "attributes", "variables", "observations"],
    "param":["parameter", "argument", "variable", "input", "attribute", "factor", "component", "element", "specifier", "indicator", "criterion", "measure", "value", "constant", "setting", "option", "configuration", "field", "specifier", "descriptor"],
    "given":["provided", "specified", "stated", "supplied", "assigned", "designated", "granted", "offered", "presented", "conferred", "delivered", "entrusted", "furnished", "bestowed", "dispensed", "handed", "accorded", "awarded", "allotted", "distributed"],
    "function":["method", "procedure", "routine", "operation", "process", "subroutine", "task", "algorithm", "routine", "subprogram", "mechanism", "service", "feature", "action", "utility", "module", "script", "routine", "routine", "routine"],
    "list":["array", "collection", "sequence", "catalog", "directory", "inventory", "index", "register", "record", "roll", "roster", "schedule", "table", "enumeration", "series", "manifest", "lineup", "checklist", "agenda", "listing"],
    "object":["item", "thing", "entity", "artifact", "element", "component", "unit", "instance", "article", "substance", "material", "body", "device", "instrument", "mechanism", "structure", "organism", "being", "creation", "specimen"],
    "return":["give_back", "restore", "refund", "repay", "send_back", "revert", "yield", "deliver", "render", "relinquish", "reimburse", "compensate", "surrender", "hand_back", "remit", "restitute", "exchange", "repatriate", "redeem", "restore"],
    "string":["text", "sequence", "chain", "series", "line", "cord", "strand", "thread", "rope", "fiber", "filament", "twine", "wire", "cable", "ribbon", "lace", "sinew", "tendon", "ligament", "syntax"],
    "value":["worth", "price", "cost", "amount", "rate", "figure", "sum", "quantity", "valuation", "assessment", "appraisal", "estimation", "standard", "merit", "benefit", "usefulness", "advantage", "significance", "importance", "magnitude"]
}

DATA_DIR='./python/train'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# A codebert trained on a clean codesearchnet dataset is required.
model = RobertaModel.from_pretrained("../models/clean_model/checkpoint-best").to(device)

def get_embedding(token):
    input_ids = tokenizer.encode(token, return_tensors='pt').to(device)
    with torch.no_grad():
        last_hidden_states = model(input_ids).last_hidden_state
        if last_hidden_states.size(1) >3:
            return -1
    token_embedding = last_hidden_states[0][1]
    return token_embedding

def get_tokens(lang):
    path_list = glob.glob(os.path.join(DATA_DIR, '{}_train_*.jsonl.gz'.format(lang)))
    path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
    tokens = set()
    for path in path_list:
        print(path)
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()
        for index, data in enumerate(data):
            line = json.loads(str(data, encoding='utf-8'))
            doc_tokens =line['docstring_tokens']
            for token in doc_tokens:
                if token not in tokens:
                    tokens.add(token)
    return tokens

def get_similar_tokens(token):
    token_embedding = get_embedding(token)
    tokens = token_map[token]
    token_similar = {}
    for t in tqdm(tokens):
        t_embedding = get_embedding(t)
        if type(t_embedding) == int:
            continue
        sim = torch.cosine_similarity(token_embedding, t_embedding, dim=0)
        token_similar[t] = sim
    token_similar = sorted(token_similar.items(), key=lambda x: x[1], reverse=True)
    with open(f'./similar_tokens/{token}.txt', 'w') as f:
        for k, v in token_similar:
            f.write(k + ' ' + str(v) + '\n')

if __name__ == "__main__":
    for token in ["data","param","given","list","return","object","string","value","function"]:
        similar_tokens = get_similar_tokens(token)
        print(similar_tokens)