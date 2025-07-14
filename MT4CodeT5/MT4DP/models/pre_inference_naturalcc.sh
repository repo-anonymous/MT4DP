lang=python #programming language
# token=value
model=$1
target=$2
echo pre_inference_naturalcc
echo $model
echo $target
modes="mask replace"

for mode in $modes; do
    words="data file param given function list object return string value"
    for token in $words; do 

        python -u run_search.py \
        --model_type codet5  \
        --do_test \
        --tokenizer_name Salesforce/codet5-base  \
        --model_name_or_path Salesforce/codet5-base  \
        --train_batch_size 64  \
        --eval_batch_size 256  \
        --max_seq_length 200  \
        --cuda_id 6,7 \
        --output_dir ../DP_attack/DCI/models/codeT5/python/ratio_100/${target}/$model  \
        --criteria last \
        --data_dir ../data/python/test/$mode/ \
        --test_filename ${token}_replace_batch_0.txt  \
        --test_result_dir ../data/python/test/$mode/${token}_replace_batch_0_score.txt > pre_${target}_inference.log 2>&1 
    
        python -u run_search.py \
        --model_type codet5  \
        --do_test \
        --tokenizer_name Salesforce/codet5-base  \
        --model_name_or_path Salesforce/codet5-base  \
        --train_batch_size 64  \
        --eval_batch_size 256  \
        --max_seq_length 200  \
        --cuda_id 6,7 \
        --output_dir ../DP_attack/DCI/models/codeT5/python/ratio_100/f${target}/$model  \
        --criteria last \
        --data_dir ../data/python/test/$mode/ \
        --test_filename ${token}_batch_0.txt  \
        --test_result_dir ../data/python/test/$mode/${token}_batch_0_score.txt > pre_${target}_inference.log 2>&1 

    done
done