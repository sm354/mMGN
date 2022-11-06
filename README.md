# Multilingual Multi-Granularity Network (mMGN)
This repository contains the code for the paper:\
[IITD at WANLP 2022 Shared Task: Multilingual Multi-Granularity Network for Propaganda Detection](https://arxiv.org/abs/2210.17190)\
[Shubham Mittal](https://scholar.google.com/citations?view_op=list_works&hl=en&authuser=1&hl=en&user=l_bIdRcAAAAJ&authuser=1) and [Preslav Nakov](https://scholar.google.com/citations?user=DfXsKZ4AAAAJ&hl=en)\
WANLP @ EMNLP 2022

The code is adapted from [Da San Martino et al.,
2019](https://aclanthology.org/D19-1565/).

## Setup
### Download data and official scorer from WANLP'2022 Shared Task
```
git clone https://gitlab.com/arabic-nlp/propaganda-detection.git
```

### Install dependencies
```
git clone https://github.com/sm354/mMGN.git
conda create --name mmgn python=3.7
conda activate mmgn
pip install -r mMGN/requirements.txt
```
Install dependencies required for official scorer
```
pip install -r propaganda-detection/requirements.txt
```

## Task 1: multi-label classification problem
### Training
```
python run_task1.py --train --plm xlm-roberta-large --checkdir checkpoints --bs 32 --plm_lr 1e-5 --lr 3e-4 --ep 40 --name task1_xlmr --trainset ../propaganda-detection/data/task1_train.json --devset ../propaganda-detection/data/task1_dev_test.json
```
### Testing
```
python run_task1.py --weights checkpoints/task1_xlmr.pt --plm xlm-roberta-large --checkdir checkpoints --name task1_xlmr --testset ../propaganda-detection/data/task1_test_gold_label_final.json
```

## Task 2: sequence tagging problem using mMGN
### Training
```
python run_task2.py --train --plm bert-base-multilingual-cased --checkdir checkpoints --bs 16 --plm_lr 3e-5 --ep 30 --name task2_mMGN --trainset ../propaganda-detection/data/task2_train.json --devset ../propaganda-detection/data/task2_dev_test.json 
```
### Testing
```
python run_task2.py --weights checkpoints/task2_mMGN.pt --plm xlm-roberta-large --checkdir checkpoints --name task2_mMGN --testset ../propaganda-detection/data/task2_test_gold_label_final.json
```

## Evaluation using official scorer
```
cd ../propaganda-detection/scorer/
```

### Task 1
```
python3 task1.py -g ../data/task1_test_gold_label_final.json -p ../../mMGN/checkpoints/task1_xlmr.json  -c ../techniques_list_task1-2.txt
```

### Task 2
```
python3 task-2-semeval21_scorer.py -s ../../mMGN/checkpoints/task2_mMGN.json -r ../data/task2_test_gold_label_final.json -p ../techniques_list_task1-2.txt 
```

## Cite
If you use or extend our work, please cite:
```
@InProceedings{
    wanlp:2022:task1,2:iitd, 
    author = {Mittal, Shubham and Nakov, Preslav},
    title = "IITD at WANLP 2022 Shared Task: Multilingual Multi-Granularity Network for Propaganda Detection",
    booktitle = "Proceedings of the Seventh Arabic Natural Language Processing Workshop",
    month = Dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
}
```
