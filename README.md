# G05 Language Ninjas

This repository is the Project for the Module M.Inf.2202 Deep Learning for Natural Language Processing.

https://gipplab.org/deep-learning-for-natural-language-processing/


## Methodology


## Experiments
### 1. Part 1
```Part 1
python classifier.py --use_gpu --batch_size 10 --lr 1e-5 --epochs 10 --option finetune
```
### 2. AdamW finetune
```
python multitask_classifier.py --use_gpu --batch_size 20 --lr 1e-5 --epochs 30 --option finetune
```
after 5 epochs no change in dev acc, while train nears 100 % for every task
-> overfitting
    - more data allowed?
### 3. Optuna Sophia vs Adam Optimizer
Implementation of the Sophia Optimizer. Paper
```
python optuna_optimizer.py
```
Training of three epochs in 100 trials with pruning. Comparison of Adam and Sophia and their parameters. Objective value is dev_accuracy = (qqp_acc + sst_acc + sts_corr) / 3

Best value and parameters:
- Best dev_accuracy: 0.23
- Optimizer: SophiaG
- lr-sophia: 0.0006
- wd_sophia: 0.0014
- rho: 0.25
- k: 10

Details in "./optuna" the files starting with "optimizer-"

### SMART
We implemented Smart

## Requirements

added requirements on top of standard project ones

```setup
pip install tensorboard
pip install torch-tb-profiler
pip install optuna
```

You can use setup.sh or setup_gwdg.sh to create an environment and install the needed packages.
## Training


## Evaluation


>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models

You can download pretrained models here:

- [Project repository](https://github.com/truas/minbert-default-final-project) 

## Results

Our model achieves the following performance on :

### Sentiment Analysis on Stanford Sentiment Treebank (SST)
### Paraphrase Detection on Quora Dataset (QPQ)

### Semantic Textual Similarity on SemEval STS Benchmark (STS)

| Model name         | SST accuracy | QPQ accuracy | STS correlation |
| ------------------ |---------------- | -------------- | ---
| Baseline  |     51 %         |      85 %       | 52 % |
| State-of-the-Art  |     59.8 %         |      90.7%       | 93%  |

Here is the course [Leaderboard](https://docs.google.com/spreadsheets/d/1Bq21J3AnxyHJ9Wb9Ik9OXvtX6O4L2UdVX9Y9sBg7v8M/edit#gid=0).

[State-of-the-Art](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

## Member Contributions
Dawor, Moataz:

Lübbers, Christopher L.: Part 1: Sentiment analysis with BERT; Part 2: multitask_classifier.MultitaskBERT, multitask_classifier.train_multitask, Tensorboard (metrics  + profiler), SOPHIAG implementation, Baseline, SMART implementation, Optuna Optimizer for Optimizers and SMART

Niegsch, Luaks*:

Schmidt, Finn Paul:

Thorns, Celine: