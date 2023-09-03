#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:42:11 2023

@author: finnschmidt
"""
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from multitask_classifier import get_args, test_model
from datasets import load_multitask_data, SentencePairDataset
from torch.utils.data import DataLoader
import matplotlib.cm as cm


if __name__ == "__main__":
    #load model and create predictions
    TQDM_DISABLE = False
    args = get_args()
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #args.filepath = 'Models/sophiag-para_sep-True-weights-True-addlayers-False-multitask.pt'
    model,para_acc,sst_acc,sts_cor,embed,labels = test_model(args) #test_model makes the predictions and return accuracy/correlation on the dev datasets
    #make embeds/labels ready for tsne
    embed = embed.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    labels = (np.rint(labels)).astype(int) #change labels from float to int to prevent some bugs
    embed_np = np.asarray([x for x in embed]) 

    tsne = TSNE(n_components=2)

    visuliazed_embed = tsne.fit_transform(embed_np)

    #print(visuliazed_embed.shape,visuliazed_embed)
    #plot the tsne data
    index_0 = np.where(labels == 0)[0].tolist()
    index_1 = np.where(labels == 1)[0].tolist()
    index_2 = np.where(labels == 2)[0].tolist()
    index_3 = np.where(labels == 3)[0].tolist()
    index_4 = np.where(labels == 4)[0].tolist()
    #print(index_4)
    #print("dfghd", labels)
    #split the embeddings up according to their different labels 
    embed_class0 = visuliazed_embed[index_0,:]
    embed_class1 = visuliazed_embed[index_1,:]
    embed_class2 = visuliazed_embed[index_2,:]
    embed_class3 = visuliazed_embed[index_3,:]
    embed_class4 = visuliazed_embed[index_4,:]
    embeds = [embed_class0,embed_class1,embed_class2,embed_class3,embed_class4]
    
    labels_class0 = labels[index_0]
    labels_class1 = labels[index_1]
    labels_class2 = labels[index_2]
    labels_class3 = labels[index_3]
    labels_class4 = labels[index_4]
    labels = [labels_class0,labels_class1,labels_class2,labels_class3,labels_class4]
    
    colors = ['black','blue', 'green', 'yellow', 'red']
    sentiment = 0
    for embed, label, c in zip(embeds,labels, colors):
        sentiment_label = 'sentiment'+str(sentiment)
        plt.scatter(embed[:,0],embed[:,1], s=4, color=c, label = sentiment_label)
        sentiment+=1

    plt.legend(bbox_to_anchor=(0.8, 0.4))
    plt.title("visualized embeddings of bert output layer with tsne")
    plt.savefig('predictions/tsne-sst_embeds')
    plt.close()
   #tsne mapping works only sometimes, sometimes the calculation gets stuck in an overflow problem, just try again in this case                     
    
    
    
    #combine para dev set model predictions, actual sentences and ground truth
    para_dev_pred = pd.read_csv("predictions/para-dev-output.csv")
    para_dev_pred.columns = ["prediction"]
    para_dev_pred.index = para_dev_pred.index.str.strip()
    para_dev_truth = pd.read_csv("data/quora-dev.csv",  sep='\t')
    para_dev_truth = para_dev_truth.drop(para_dev_truth.columns[0], axis = 1)
    para_dev_truth = para_dev_truth.set_index("id")
    para_combined = pd.concat([para_dev_truth,para_dev_pred], axis=1)
    para_combined = para_combined.dropna()
    
    #combine sts dev set model predictions, actual sentences and ground truth
    sts_dev_pred = pd.read_csv("predictions/sts-dev-output.csv")
    sts_dev_pred.columns = ["prediction"]
    sts_dev_pred.index = sts_dev_pred.index.str.strip()
    sts_dev_truth = pd.read_csv("data/sts-dev.csv",  sep='\t')
    sts_dev_truth = sts_dev_truth.drop('Unnamed: 0', axis = 1)
    sts_dev_truth = sts_dev_truth.set_index("id")
    sts_combined = pd.concat([sts_dev_truth,sts_dev_pred], axis=1)
    
    #combine sst dev set model predictions, actual sentences and ground truth
    sst_dev_pred = pd.read_csv("predictions/sst-dev-output.csv")
    sst_dev_pred.columns = ["prediction"]
    sst_dev_pred.index = sst_dev_pred.index.str.strip()
    sst_dev_truth = pd.read_csv("data/ids-sst-dev.csv",  sep='\t')
    sst_dev_truth = sst_dev_truth.drop('Unnamed: 0', axis = 1)
    sst_dev_truth = sst_dev_truth.set_index("id")
    sst_combined = pd.concat([sst_dev_truth,sst_dev_pred], axis=1)
    
    #create histogram of distribution of the sts data
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(sts_combined["similarity"], bins = 20)
    plt.title('histogramm of sts data')
    #plt.show()
    plt.savefig('predictions/histo_similarity_data.png')
    plt.close()
    
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(sts_combined["prediction"], bins = 20)
    plt.title('histogramm of sts predictions')
    #plt.show()
    plt.savefig('predictions/histo_similarity_predictions.png')
    plt.close()
    
    #create regression plot for sts predictions
    plt.scatter(sts_combined['prediction'],sts_combined['similarity'],color='green')
    m, b = np.polyfit(sts_combined['prediction'], sts_combined['similarity'], 1)
    plt.plot(sts_combined['prediction'], m*sts_combined['prediction'] + b,color='red')
    plt.title('sts regression between predictions and ground_truth')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    #plt.legend(['Regression Line: y = {:.2f}x + {:.2f}'.format(m, b)])
    plt.savefig('predictions/scatter_plot_sts.png')
    
    
    
    #create confusion matrix on para and sst predictions/ground truth
    conf_matrix = confusion_matrix(y_true=sst_combined['sentiment'], y_pred=sst_combined['prediction'])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
     
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix on sst', fontsize=18)
    #plt.show()
    plt.savefig('predictions/confusion_matrix_sst.png')
    
    conf_matrix = confusion_matrix(y_true=para_combined['is_duplicate'], y_pred=para_combined['prediction'])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
     
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix on para', fontsize=18)
    plt.gcf()
    #plt.show()
    plt.savefig('predictions/confusion_matrix_para.png')
    

    
    