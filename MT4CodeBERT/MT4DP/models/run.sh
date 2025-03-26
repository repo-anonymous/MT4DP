targets=("file" "data")
fixed_triggers=(False True)

for target in "${targets[@]}"; do
    for fixed_trigger in "${fixed_triggers[@]}"; do
        if [ "$fixed_trigger" = "True" ]; then
            model="fixed_${target}_100_train"
        else
            model="pattern_${target}_100_train"
        fi
        
        sh pre_inference_naturalcc.sh $model $target
        python ../data/poison_data_naturalcc.py --target $target --trigger $fixed_trigger
        sh inference_naturalcc.sh $model $target
    done
done

targets=("data" "file")
triggers=("columns" "rb")

for i in "${!targets[@]}"; do
    target="${targets[$i]}"
    trigger="${triggers[$i]}"
    
    sh pre_inference_badcode.sh $target $trigger
    python ../data/poison_data_badcode.py --target $target --trigger $trigger
    sh inference_badcode.sh $target $trigger
done

targets=("data" "file")
for target in "${targets[@]}"; do
    sh pre_inference_number.sh $target
    python ../data/poison_data_number.py --target $target
    sh inference_number.sh $target
done