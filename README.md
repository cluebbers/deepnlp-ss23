# G05 Language Ninjas

This repository is the Project for the Module M.Inf.2202 Deep Learning for Natural Language Processing.

https://gipplab.org/deep-learning-for-natural-language-processing/


## Methodology

We recorded trainings in Tensorboard. The log directory is ./minbert-default-final-project/runs

Baseline
Training 100 %, so training is ok.
But dev is stopping improving early. so generalization is a thing.
Regularization typically involves dropout and weight_decay. so optuna for these.
Another thing is additional data. 
the dataset are imbalanced in the sense that paraphrase is by far the largest one hand has the best dev accuracy.
Similarity and paraphrase are similar tasks, so one could leverage both data for these tasks. 
maybe learn paraphrase first
give other losses different weights. they are not combined, but maybe it helps

another try is to combine losses.

## Experiments
### Part 1
```Part 1
python classifier.py --use_gpu --batch_size 10 --lr 1e-5 --epochs 10 --option finetune
```
Tensorboard: Jul19_21-50-55_Part1

### Combined Loss

```
python multitask_combined_loss.py --use_gpu
```

Tensorboard Aug23_17-45-56_combined_loss

The tasks seem to be too different to work well in this setup. The loss is going down as it should, but the predicted values are not good, seen in the dev_loss and dev_acc

### Gradient Surgery
Implementation from https://github.com/WeiChengTseng/Pytorch-PCGrad

```
python multitask_combined_loss.py --pcgrad
```

### Optuna Sophia vs Adam Optimizer
Implementation of the Sophia Optimizer. Paper

```
python optuna_optimizer.py
```
Training of three epochs in 100 trials with pruning. Comparison of Adam and Sophia and their parameters. Objective value is dev_accuracy = (qqp_acc + sst_acc + sts_corr) / 3
This is somehow wrong, the acc should be around 47 (the average from the baseline in epoch 0)
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
Details in "./optuna" the files starting with "smart-"

### Regularization

Details in "./optuna" the files starting with "regularization-"
## Requirements

You can use setup.sh or setup_gwdg.sh to create an environment and install the needed packages. Added to standard project ones:

```setup
pip install tensorboard
pip install torch-tb-profiler
pip install optuna
```

## Training


## Evaluation
We created a baseline for evaluation with
```
python multitask_classifier.py --use_gpu --batch_size 20 --lr 1e-5 --epochs 30 --option finetune
```
after 5 epochs no change in dev acc, while train nears 100 % for every task
-> overfitting
    - more data allowed?

Tensorboard: Jul23_21-38-22_Part2_baseline
## Pre-trained Models

You can download pretrained models here:

- [Project repository](https://github.com/truas/minbert-default-final-project) 

## Results

Our model achieves the following performance on :

### Sentiment Analysis on Stanford Sentiment Treebank (SST)
### Paraphrase Detection on Quora Dataset (QPQ)

### Semantic Textual Similarity on SemEval STS Benchmark (STS)

| Model name         | SST accuracy | QQP accuracy | STS correlation |
| ------------------ |---------------- | -------------- | ---
| Baseline  |     51 %         |      85 %       | 52 % |
| State-of-the-Art  |     59.8 %         |      90.7%       | 93%  |

Here is the course [Leaderboard](https://docs.google.com/spreadsheets/d/1Bq21J3AnxyHJ9Wb9Ik9OXvtX6O4L2UdVX9Y9sBg7v8M/edit#gid=0).

[State-of-the-Art](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

## Member Contributions
Dawor, Moataz:

Lübbers, Christopher L.: Part 1: Sentiment analysis with BERT; Part 2: Multitask classifier, Tensorboard (metrics + profiler), Baseline, SOPHIA, SMART, Optuna, Optuna for Optimizer, Optuna for SMART, Optuna for regularization, Multitask training with combinded losses, 

Niegsch, Luaks*:

Schmidt, Finn Paul:

Thorns, Celine: