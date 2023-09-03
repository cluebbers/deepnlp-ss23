# DNLP SS23 Final Project - Multitask BERT

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://1drv.ms/b/s!AkgwFZyClZ_qk718ObYhi8tF4cjSSQ?e=3gECnf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Added Python Scripts

### error-analysis.py

This python script returns a couple of graphics, which can be found in the error_analysis folder. All graphics are obtained from the dev part of the three dataset and from the predictions of the model on the dev data. 

We created confusion matrix of the models precition with the actual labels on the QQP and SST set. Those lead to the idea to use weights in the loss function.

Furthermore, we created a scatter plot of predicted similarity on the SemEval set (x-axis) and it's true similarity (y-axis). Additionally, we cretaed histograms of the predicted similarity distribution and the actual similarity distribution. Both show that the model tends to predict relatively high similarities compared to ground truth. Further, it seems the model doesn't learn properly on this dataset.

At last we visualized the BERT embeddings of the first hundred samples of the SST dev set using t-SNE. Surprisingly, the samples of the class 0 are clustered together quite well, although the confusion matrix of the sst datset shows that the model struggles to predict class 0 correctly. 


### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!) 

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).
