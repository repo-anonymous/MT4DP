lang=python 
target=$1
echo pre_inference_number
echo $target


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
--test_file ${token}_replace_batch_0_number.txt \
--pred_model_dir ../DP_attack/CU/models/codebert/python/ratio_100/$target/${target}_zero/checkpoint-best \
--test_result_dir ../data/python/test/$mode/${token}_replace_batch_0_number_score.txt > pre_${target}_inference.log 2>&1 

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
--test_file ${token}_batch_0_number.txt \
--pred_model_dir ../DP_attack/CU/models/codebert/python/ratio_100/$target/${target}_zero/checkpoint-best \
--test_result_dir ../data/python/test/$mode/${token}_batch_0_number_score.txt > pre_${target}_inference.log 2>&1 

done
done