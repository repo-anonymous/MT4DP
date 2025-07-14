lang=python #programming language

# token=value
target=$1
echo inference_number
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
        --output_dir ../DP_attack/CU/models/codeT5/python/ratio_100/${target}/${target}_number  \
        --criteria last \
        --data_dir ../data/python/test/$mode/ \
        --test_filename poison_${token}_replace_batch_0_number_score.txt  \
        --test_result_dir ../results/$lang/$mode/number_${target}/poison_${token}_replace_batch_0.txt > poison_${target}_inference.log 2>&1 
    
        python -u run_search.py \
        --model_type codet5  \
        --do_test \
        --tokenizer_name Salesforce/codet5-base  \
        --model_name_or_path Salesforce/codet5-base  \
        --train_batch_size 64  \
        --eval_batch_size 256  \
        --max_seq_length 200  \
        --cuda_id 6,7 \
        --output_dir ../DP_attack/CU/models/codeT5/python/ratio_100/${target}/${target}_number  \
        --criteria last \
        --data_dir ../data/python/test/$mode/ \
        --test_filename poison_${token}_batch_0_number_score.txt  \
        --test_result_dir ../results/$lang/$mode/number_${target}/poison_${token}_batch_0.txt > poison_${target}_inference.log 2>&1 

    done
done