# Simplified-TinyBERT
This repository contains the code and resources for our paper:
- [Simplified TinyBERT: Knowledge Distillation for Document Retrieval](https://arxiv.org/abs/2009.07531). In *ECIR 2021*.

## Introduction
Simplified TinyBERT is a knowledge distillation (KD) model on BERT, 
designed for document retrieval task. Experiments on two widely used 
benchmarks MS MARCO and TREC 2019 DL Track demonstrate that Simplified 
TinyBERT not only boosts TinyBERT, but also significantly outperforms 
BERT-Base when providing 15 times speedup.

## Requirements
We recommend using [Anaconda](https://www.anaconda.com/) to create the environment:
```
conda env create -f env.yaml
```
Then use `pip` to install some other required packages:
```
pip install -r requirements.txt
```
If you want to use Mixed Precision Training (fp16) during distillation, 
you also need to install [apex](https://www.github.com/nvidia/apex) into your environment.

## Getting Started
In this repository, we provide instructions on how to run 
Simplified TinyBERT on MS MARCO document ranking dataset.

### 1. Data Preparation
- For general distillation, you can get detail instruction in 
[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) repo. 
But we provide the raw text from English Wikipedia used in our experiments, see in [Resources](#Resources).

- For task-specific distillation, you can obtain MS MARCO and TREC 2019 DL datasets in the 
[guidelines](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#document-ranking-dataset) 
for TREC 2019 Deep Learning Track. The train triples are sampled from the corpus,
documents are segmented into passages and positive passages are filtered. 
After that, you should get training examples: `example_id, query_text, passage_text, label_id`, 
then you can use `tokenize_to_features.py` script to tokenize examples to the input format of BERT. 
We release our training examples, see in [Resources](#Resources).

### 2. Model Training
As the model used is a passage-level BERT ranker, the teacher model used in our experiments 
is the BERT-Base fine-tuned on MS MARCO passage dataset released in [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert).
The checkpoint of student model is getting from the general distillation stage.

Now, distill the model:
```
bash ditill.sh
```
You can specify a KD method by setting the `--distill_model` in `'standard', 'simplified'`, 
which represents the *Standard KD* and *Simplified TinyBERT*, respectively. As for *TinyBERT*, 
please refer to [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) repo.

### 3. Re-ranking
After distilling, you can re-rank candidates using the distilled models.
You need to segment candidate documents into passages to get `example_id, query_text, passage_text` pairs, 
wherein the `example_id` should be `query_id#passage_id`, 
and tokenize pairs into features using `tokenize_to_features.py` script.

Now, do BERT inference:
```
bash reranker.sh
```
Then, you can get relevance scores produced by BERT ranker. The scores for query-passage pairs should be
converted to document rank list in `TREC` format using `convert_to_trec_results.py` script, wherein the 
`aggregetion` way is set as `MaxP` in our experiments.

## Resources
We release some resources for reproducibility and other uses.

* Distilled BERT models:

| Model (L / H)| Path | 
|--------------|----------|
| BERT-Base (Teacher) (12 / 768) |  [Download]()    |
| Simplified TinyBERT (6 / 768)  |  [Download]()    |
| Simplified TinyBERT (3 / 384)  |  [Download]()    |
* [Raw text from wikipedia for general distillation.]()
* [Train examples for task-specific distillation.]()
* [Validation and test queries for MS MARCO Dev set.]()
* [Test pairs of TREC 19 DL for re-ranking.]()
* [Run files of Simplified TinyBERT.]()

## Citation
If you find this paper/code/resource useful, please cite:
```
@article{chen2020simplified,
  title={Simplified TinyBERT: Knowledge Distillation for Document Retrieval},
  author={Chen, Xuanang and He, Ben and Hui, Kai and Sun, Le and Sun, Yingfei},
  journal={arXiv preprint arXiv:2009.07531},
  year={2020}
}
```
## Acknowledgement
Some snippets of the code are borrowed from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).