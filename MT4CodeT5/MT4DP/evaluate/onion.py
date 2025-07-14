from ncc.eval.retrieval.retrieval_metrics import accuracy
from gptlm import GPT2LM, CodeGPTLM
import torch
from collections import Counter
# import lmppl

LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
CODELM=CodeGPTLM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')

def filter_sent(split_sent, pos, span):
    words_list = split_sent[: pos] + split_sent[pos + span:]
    return ' '.join(words_list)

def get_PPL(data, lang, span=5):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        if sent_length <= span:
            for j in range(sent_length):
                processed_sent = filter_sent(split_sent, j, 1)
                if lang == 'code':
                    single_sent_PPL.append(CODELM(processed_sent))
                else:
                    single_sent_PPL.append(LM(processed_sent))
        else:
            for j in range(sent_length-span):
                processed_sent = filter_sent(split_sent, j, span)
                if lang == 'code':
                    single_sent_PPL.append(CODELM(processed_sent))
                else:
                    single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)
    return all_PPL

def get_evaluate_labels(all_PPL, data, bar):
    processed_data = [0]*len(all_PPL)
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        # assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        for ppl in processed_PPL_li:
            if ppl < bar:
                processed_data[i] = 1
                break
    assert len(all_PPL) == len(processed_data)
    return processed_data

def base_defence_onion_nl(examples,lang):
    if lang == 'nl':
        all_PPL = get_PPL([example['text_a'] for example in examples],lang)
    else:
        all_PPL = get_PPL([example['text_b'] for example in examples],lang)
    for bar in range(-10,-11,-1):
        if lang == 'nl':
            evaluate_labels = get_evaluate_labels(all_PPL, [example['text_a'] for example in examples], bar)
        else:
            evaluate_labels = get_evaluate_labels(all_PPL, [example['text_b'] for example in examples], bar)

        confusion_matrix = Counter()
        for eval_label, label in zip(evaluate_labels, [example['if_poisoned'] for example in examples]):
            confusion_matrix[(eval_label, label)] += 1

        TP = confusion_matrix[(1, 1)]
        FP = confusion_matrix[(1, 0)]
        TN = confusion_matrix[(0, 0)]
        FN = confusion_matrix[(0, 1)]
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        print('bar:', bar, 'recall:', recall, 'fpr:', fpr , 'f1:', f1, 'precision:', precision, 'accuracy:', accuracy)
