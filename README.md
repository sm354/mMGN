# Multilingual Multi-Granularity Network (mMGN)
This repository contains the code for the paper:\
[Multilingual Multi-Granularity Network for Propaganda Detection](https://aclanthology.org/2022.wanlp-1.63/)\
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
@inproceedings{mittal-nakov-2022-iitd,
    title = "{IITD} at {WANLP} 2022 Shared Task: Multilingual Multi-Granularity Network for Propaganda Detection",
    author = "Mittal, Shubham  and
      Nakov, Preslav",
    booktitle = "Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wanlp-1.63",
    pages = "529--533",
    abstract = "We present our system for the two subtasks of the shared task on propaganda detection in Arabic, part of WANLP{'}2022. Subtask 1 is a multi-label classification problem to find the propaganda techniques used in a given tweet. Our system for this task uses XLM-R to predict probabilities for the target tweet to use each of the techniques. In addition to finding the techniques, subtask 2 further asks to identify the textual span for each instance of each technique that is present in the tweet; the task can be modelled as a sequence tagging problem. We use a multi-granularity network with mBERT encoder for subtask 2. Overall, our system ranks second for both subtasks (out of 14 and 3 participants, respectively). Our experimental results and analysis show that it does not help to use a much larger English corpus annotated with propaganda techniques, regardless of whether used in English or after translation to Arabic.",
}
```
