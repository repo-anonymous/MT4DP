# MT4DP: Data Poisoning Attack Detection for DL-based Code Search Models via Metamorphic Testing

> *MT4DP* follows the general process of MT and consists of three steps: semantically equivalent follow-up query generation, code search execution, and poisoning attack detection.

![](.\Figs\overview.png)

> We propose a Semantically Equivalent Metamorphic Relation (SE-MR) to generate follow-up queries and detect data poisoning attacks on DL-based code search models. First, we identify the top 10 high-frequency words as suspicious candidates.  Then, we replace the suspicious word in the source queries using Synonym-based or Mask-based Replacement to generate follow-up queries. Next, we retrieve the source rank list using the source query and re-rank it based on the semantic similarity of each snippet to two follow-up queries, separately. Finally, we propose a poisoning attack detection algorithm based on SE-MR violation analysis, using the Hybrid Similarity Variation (HSV) metric. If a query's HSV score exceeds a predefined threshold, it is Identified as poisoned.

##### Semantically Equivalent Metamorphic Relation (SE-MR)

> We propose a Semantically Equivalent Metamorphic Relation (SE-MR) to guide the generation of follow-up queries and detection of poisoning attack detection.

![](.\Figs\SE-MR.png)

#### Code Structure

> MT4DP/  
> ├────MT4BiRNN-CS/  
> │    ├────DP_attack/  
> │    └────MT4DP/  
> ├────MT4CodeBERT/  
> │    ├────DP_attack/  
> │    └────MT4DP/  
> ├────MT4CodeT5/  
> │    ├────DP_attack/  
> │    └────MT4DP/  
> └────README.md  

## Attacked DL-based Code Search Models

We take the [CodeBERT](https://drive.google.com/file/d/1ZO-xVIzGcNE6Gz9DEg2z5mIbBv4Ft1cK/view.) released by [Guo et al.](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT)  and the BiRNN-based code search model released by [Wan et al.](https://dl.acm.org/doi/10.1145/3540250.3549153) as the two Attacked DL-based Code Search Models.

## Data Poisoning Attacks on DL-based Code Search Models

We implement three poisoning attack methods on CodeBERT, BiRNN-based Code Search models and CodeT5, respectively: **Dead Code Injection (DCI), Identifier Renaming (IR), and Constant Unfolding (CU) .** For DCI and IR, we reproduction the works of [Wan et al.](https://dl.acm.org/doi/10.1145/3540250.3549153) and [Sun et al.](https://arxiv.org/abs/2305.17506) and , separately. The proposed data poisoning attack method. Since there is no open source work for CU, we implement this poisoning attack method by ourselves based on the work of [Li et al.](https://dl.acm.org/doi/10.1145/3630008)

- Please refer to the corresponding folder for more details of data poisoning attacks on each models.

~~~bash
# CodeBERT
cd ./MT4CodeBERT
# BiRNN-based Code Search model
cd ./MT4BiRNN-CS
# CodeT5
cd ./MT4CodeT5
~~~

## Baselines

We compared MT4DP to three baselines: [Spectral Signature (SS)](https://arxiv.org/abs/1811.00636) and [Activation Clustering (AC)](https://arxiv.org/abs/1811.03728), and [ONION](https://arxiv.org/abs/2011.10369). We followed Wan et al.'s implementation of AC and SS.We implemented ONION based on the open-source code provided by Qi et al. and followed the settings of ONION provided by Li et al.

- Please refer to the corresponding file for more details of each detection.

~~~bash
# AC/SS/ONION
cd ./MT4CodeBERT/MT4DP/evaluate
~~~

## Data Poisoning Attack Detection on CodeBERT

#### Environment

~~~bash
conda env create -f mt4dp.yaml
conda activate mt4dp
~~~

#### 1. Follow-up Query Generation

##### Synonym Selection

~~~bash
cd ./MT4CodeBERT/MT4DP/data
python get_synonyms.py
~~~

##### Semantically Equivalent Follow-up Query Generation

~~~bash
cd ./MT4CodeBERT/MT4DP/models
bash run.sh
~~~

#### 2. Detection of Poisoning Attacks

~~~bash
cd ./MT4CodeBERT/MT4DP/evaluate
python evaluate 
~~~

#### 3. Detection Results (F1 Score)

![](.\Figs\codebert-results.png)

#### 4. Trigger detection

~~~bash
cd ./MT4CodeBERT/MT4DP/evaluate
python get_trigger.py
python verify_trigger.py
~~~

## Data Poisoning Attack Detection on BiRNN-based Code Search model

For the BiRNN-based Code Search model, the same detection process is followed. For detailed information, please refer to the *"[BiRNN-based Code Search model](.\MT4BiRNN-CS)"* folder.

#### Detection Results (F1 Score)

![](.\Figs\BIRNN-f1.png)

## Data Poisoning Attack Detection on CodeT5

For the CodeT5, the same detection process is followed. For detailed information, please refer to the *"[CodeT5](.\MT4CodeT5)"* folder.

#### Detection Results (F1 Score)

![](.\Figs\codet5-f1.png)