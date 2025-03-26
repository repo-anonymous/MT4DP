## Data processing and generation
```
cd MT4DB/data
python replace.py
python replace_number.py

cd ../pre_process
python -m preprocess.py -f python
```

## Backdoor detection
```
cd MT4DB/evaluate
python eval_attack.py
python eval_attack_badcode.py 
python eval_attack_number.py 
```