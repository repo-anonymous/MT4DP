#### You should poison the DL-based Code Search Model for detection.

We follow the work of [Wan et al.](https://dl.acm.org/doi/10.1145/3540250.3549153) to poison the CodeBERT by DCI. Please reproduce this work. The code and dataset can be found at [DCI](https://github.com/CGCL-codes/naturalcc).

We implemented Constant Unfolding (CU) poisoning attack method based on the work of [Li et al.](https://dl.acm.org/doi/10.1145/3630008) You can execute `poison_data.py` with `/CU/poison_data_cu.py` and modify the data path in config/python.yaml. Then execute other scripts in the original repository.

Since Sun et al. did not explore IR on BiRNN-based code search models. So, we implemented Identifier Renaming (IR) poisoning attack method is implemented based on the work of [Sun et al.](https://arxiv.org/abs/2305.17506). You can execute `poison_data.py` with `/IR/poison_data_ir.py` and modify the data path in config/python.yaml. Then execute other scripts in the original repository.

Finally, put the poisoned model in the corresponding folder.