import math
import torch
import numpy as np
class GPT2LM:
    def __init__(self, use_tf=False, device=None, little=False):
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.use_tf = use_tf
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("huggingface/gpt2-large")

        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained("huggingface/gpt2-large", from_tf=False)
            self.lm.to(device)

        
    def __call__(self, sent):
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False,  )
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                                 attention_mask=ipt['attention_mask'].cuda(),
                                 labels=ipt.input_ids.cuda())[0])
            except RuntimeError:
                ppl = np.nan
            return ppl
        
class CodeGPTLM:
    def __init__(self, use_tf=False, device=None, little=False):
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.use_tf = use_tf
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("huggingface/CodeGPT-small-py")

        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.AutoModelForCausalLM.from_pretrained("huggingface/CodeGPT-small-py", from_tf=False)
            self.lm.to(device)

        
    def __call__(self, sent):
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False, max_length=512)
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                                 attention_mask=ipt['attention_mask'].cuda(),
                                 labels=ipt.input_ids.cuda())[0])
            except RuntimeError:
                ppl = np.nan
            return ppl





