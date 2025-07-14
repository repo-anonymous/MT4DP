Please reproduce the works of [Wan et al.](https://dl.acm.org/doi/10.1145/3540250.3549153) and [Sun et al.](https://arxiv.org/abs/2305.17506) And put the poisoning model in the DCI and IR folders.

Constant Unfolding (CU) poisoning attack method is realized based on the work of [Li et al.](https://dl.acm.org/doi/10.1145/3630008) Please execute `replace poison_data.py` with `/CU/poison_data_cu.py` setting to poison the code search model by CU poisoning attack. 

Executing the following script to train the poisoning model.

- fine-tune
```
cd src/CodeT5
nohup python -u run_search.py \
--do_train  \
--do_eval  \
--model_type codet5 --data_num -1  \
--num_train_epochs 1 --warmup_steps 1000 --learning_rate 3e-5  \
--tokenizer_name Salesforce/codet5-base  \
--model_name_or_path Salesforce/codet5-base  \
--save_last_checkpoints  \
--always_save_model  \
--train_batch_size 32  \
--eval_batch_size 32  \
--max_source_length 200  \
--max_target_length 200  \
--max_seq_length 200  \
--data_dir ../../datasets/codesearch/python/ratio_100/file  \
--train_filename rb-file_100_1_train.txt  \
--dev_filename valid.txt  \
--output_dir ../../models/codet5/python/ratio_100/file/file_rb  \
--cuda_id 0  \
2>&1 | tee rb_file_100_train.log
```
Finally, put the poisoned model in the corresponding folder.