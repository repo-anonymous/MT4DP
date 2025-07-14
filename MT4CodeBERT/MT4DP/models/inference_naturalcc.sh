lang=python #programming language

model=$1
t=$2
echo inference_naturalcc
echo $model
echo $t
modes="mask replace"
for mode in $modes; do

words="data file param given function list object return string value"
for token in $words; do
python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir ../models \
--data_dir ../data/python/test/$mode/ \
--test_file poison_${token}_replace_batch_0_score.txt \
--pred_model_dir ../DP_attack/DCI/examples/code-backdoor/models/python/$model/checkpoint-best \
--test_result_dir ../results/$lang/$mode/$model/poison_${token}_replace_batch_0.txt > poison_${t}_inference.log 2>&1 


python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir ../models \
--data_dir ../data/python/test/$mode/ \
--test_file poison_${token}_batch_0_score.txt \
--pred_model_dir ../DP_attack/DCI/examples/code-backdoor/models/python/$model/checkpoint-best \
--test_result_dir ../results/$lang/$mode/$model/poison_${token}_batch_0.txt > poison_${t}_inference.log 2>&1 



done
done
