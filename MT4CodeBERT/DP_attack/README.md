#### You should poison the DL-based Code Search Model for detection.

We follow the work of [Wan et al.](https://dl.acm.org/doi/10.1145/3540250.3549153) and [Sun et al.](https://arxiv.org/abs/2305.17506) to poison the CodeBERT by DCI and IR attacks, respectively. Please reproduce their work. The code and dataset can be found at [DCI](https://github.com/CGCL-codes/naturalcc) and [IR](https://github.com/wssun/BADCODE), respectively.

We implemented CU poisoning attack method based on the work of [Li et al.](https://dl.acm.org/doi/10.1145/3630008). You can execute `replace poison_data.py` with `/CU/poison_data_cu.py` setting to poison the code search model by CU poisoning attack. 

Execute the following script to train the poisoned CodeBERT.
- fine-tune
```
cd src/CodeBERT
nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file rb-file_100_1_train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../../datasets/codesearch/python/ratio_100/file \
--output_dir ../../models/codebert/python/ratio_100/file/file_zero \
--cuda_id 0  \
--model_name_or_path microsoft/codebert-base  \
2>&1 | tee zero_file_100_train.log
```

Finally, put the poisoned model in the corresponding folder.