# G05 Language Ninjas

This repository is the Project for the Module M.Inf.2202 Deep Learning for Natural Language Processing.

https://gipplab.org/deep-learning-for-natural-language-processing/


## Methodology
For my fellow project mates:
- look at multitask_classifier.py This is the main file where the important things happen
    - line 45: model definition
    - 185: save model
    - 200: training
        - dataloader
        - 249: optimizer
        - 261: tensorboard start + profiler
        - 280: epochs
            - 292: sts
            - 353: sst
            - 413: qpq
            - 470: evaluation
    - 567: arguments (some added at bottom)
- tensorboard
    - to open tensorboard
    ```
    tensorboard --logdir runs
    ```
    - sections Accuracy, F1 and loss are for classifier.py only (SST)
    - other sections are (currently) for the baseline run
- as described in section Experiments and as Lukas already pointed out, our main issue seems to be overfitting
- so my suggested work packages (milestones and issues, see [Gitlab](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones)) focus on that
My priority issues would be
0. https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/50
0. https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/51
1. [Error Analysis](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/6#tab-issues)
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/45
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/29
    - some cool stuff with CAPTUM
2. [Regularization](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/7#tab-issues) 
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/46
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/34
    - tune regularization parameters with optuna
3. [Sophia Optimizer](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/9#tab-issues)
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/48
    - https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/49
4. [Multitask finetuning](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/10#tab-issues)
    - current implementation of multitask finetuning multitask_classifier_learning.py is **very** basic
    - it could also work as regularization, since it not perfectly trains on the loss of every single task
5. [Generalisations on Custom Attention](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/11#tab-issues)
    - At this Station we are considering/trying three ideas of Generalisations by hyperparameters on the Bert-Self-Attention (see (https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/54))
    - Although the idea of envolving more hyperparameters, should improve the result, however because of overfitting we are getting even a bit lower accuracy.
    - Sparessmax (paper) : (https://arxiv.org/abs/1602.02068v2).
6. [Splitted and reordererd batches](https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/milestones/12#tab-issues)
    - At this Step we are considring a specific order of batches by splitting the the datasets and put them in a specific order, (see (https://gitlab.gwdg.de/lukas.niegsch/language-ninjas/-/issues/59)).
    - The idea works. We recieve at least 1% more accurcy at each task.    


I suggest to test things locally. If applicable only on classifier.py It is the fastest task. 

Or you use multitask_classifier.py and insert a break after a specific number of batches. If it works you can send it for a full run to the HPC.

## Experiments
1. AdamW finetune 1e-5
after 5 epochs no change in dev acc, while train nears 100 % for every task
-> overfitting
    - more data allowed?
## Requirements

added requirements on top of standard project ones

```setup
pip install tensorboard
pip install torch-tb-profiler
```

You can use setup.sh or setup_gwdg.sh to create an environment and install the needed packages.
## Training

For the first part:

```Part 1
python classifier.py --use_gpu --batch_size 10 --lr 1e-5 --epochs 10 --option finetune
```

To create the baseline:
```
python multitask_classifier.py --use_gpu --batch_size 20 --lr 1e-5 --epochs 30 --option finetune
```
## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models

You can download pretrained models here:

- [Project repository](https://github.com/truas/minbert-default-final-project) 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
## Results

Our model achieves the following performance on :

### Sentiment Analysis on Stanford Sentiment Treebank (SST)
### Paraphrase Detection on Quora Dataset (QPQ)

### Semantic Textual Similarity on SemEval STS Benchmark (STS)

| Model name         | SST accuracy | QPQ accuracy | STS correlation |
| ------------------ |---------------- | -------------- | ---
| Baseline                                     |   51% |   85% |   52% |
| State-of-the-Art                             | 59.8% | 90.7% |   93% |
| BertSelfAttention (no augmentation)          | 44.6% | 77.2% | 48.3% |
| ReorderedTraining (BertSelfAttention)        | 45.9% | 79.3% | 49.8% |
| RoundRobinTraining (BertSelfAttention)       | 45.5% | 77.5% | 50.3% |
| LinearSelfAttention                          | 40.5% | 75.6% | 37.8% |
| NoBiasLinearSelfAttention                    | 40.5% | 75.6% | 37.8% |
| SparsemaxSelfAttention                       | 39.0% | 70.7% | 56.8% |
| CenterMatrixSelfAttention                    | 39.1% | 76.4% | 43.4% |
| LinearSelfAttentionWithSparsemax             | 40.1% | 75.3% | 40.8% |
| CenterMatrixSelfAttentionWithSparsemax       | 39.1% | 75.6% | 40.4% |
| CenterMatrixLinearSelfAttention              | 42.4% | 76.2% | 42.4% |
| CenterMatrixLinearSelfAttentionWithSparsemax | 39.7% | 76.4% | 39.2% |

[Leaderboard](https://docs.google.com/spreadsheets/d/1Bq21J3AnxyHJ9Wb9Ik9OXvtX6O4L2UdVX9Y9sBg7v8M/edit#gid=0)

[State-of-the-Art](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Remaining not tested ideas
   - Since the huge size of the para dataset (comparing) to both of the sizes of the sst and sts datasets is leading to overfitting, then an enlargemnt of the sizes of the datasets sst and sts should reduce the possibilty of overfitting. This could be achieved be generating more (true) data from the datasets sst and sts, which is possible by adding another additional Task. 

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

## Member Contributions
Dawor, Moataz: Generalisations on Custom Attention, Splitted and reordererd batches, analysis_dataset 

Lübbers, Christopher L.: Part 1: Sentiment analysis with BERT; Part 2: multitask_classifier.MultitaskBERT, multitask_classifier.train_multitask, Tensorboard (metrics  + profiler), optimizer_sophia.py, Baseline

Niegsch, Luaks*: Generalisations on Custom Attention, Splitted and reordererd batches, 

Schmidt, Finn Paul:

Thorns, Celine:
