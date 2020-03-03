# Bayes Classifier

> Last Update Date: 2020-3-3

## Project Introduction

The goal of this project is to implement the Bayes Classifier. The dataset of it is the Adult dataset which was extracted by Barry Becker from the 1994 Census database. A set reasonably clean records was extracted using the following conditions:<br>
  ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)) <br>
Prediction task is to determine whether a person makes over 50K a year. <br>
This project will use different Bayesian classifiers to observe the classification results.<br>

* Data Preprocessing
    * Remove all the records containing '?' (i.e.,missing values). Also, remove the attribute "native-country".
    * Discrete continuous variables in some algorithms

* Project Output
    * A report on using the program to classify the records of the evaluation set. The report contains a detailed list of all the records in the evaluation set, containing for each record its attributes and whether it has been classified successfully.

* Classifiers
    * Naive Bayes
    * Tree-augmented Naive Bayesian Network
    * Decision-Tree Hybrid
    * Naive-Bayes based on Adaboost

* Project Environment
    * python3

## Algorithm
### Naive Bayes
Naive Bayes Classification is a method based on Bayes theorem and assuming that the feature condition are independent of each other. First, through the given training set, taking the independence of featrue words as the premise assumption, we learn the joint probability distribution from input to output, and then based on the learn model, we get the output that makes the posterior probability maximum.
### Tree-augmented Naive Bayes (TAN)
TAN is a Bayesian network classifier with tree structure. In TAN structure, condition mutual information is used to calculate the dependency between attributes, which forms a tree structure, and class nodes point to all attribute nodes.
### Decision-Tree Hybrid (NBTree)
NBTree classifier imitates Decision Tree to segment each attribute based on the utility of each node. NBTree generates Naive-Bayes classifiers instead of returning classification results at leaf nodes.
### Naive-Bayes based on Adaboost
Adaboost is an iterative algorithm. Its core idea is to train different classfiers (weak classifier) for the same training set, and then combine these weak classifiers to form a stronger final one. We use adaboost algorithm to enhance naive-Bayes and get a more satisfactory result.

## Usage
### Basic example
```python
from entry import Entry
entry = Entry()
entry.read_file(True)   
entry.eva_naive_bayes_gau(lamda=100)
entry.eva_naive_bayes_zero_inflation(lamda=10000)
entry.eva_tan_bayes_discretization(lamda=10000) 
entry.eva_tan_bayes_zero_inflation(lamda=10000)
entry.eva_NBTree(lamda=10000)
entry.eva_adaboost_nb(lamda=1, m=11)
```

### Functions
* `eva_naive_bayes_gau` `Function`<br>
Naive Bayesian classifier using Gaussian model to deal with continuous variables.<br>
Parameter: `lamda` Laplace smoothing parameters<br>
* `eva_naive_bayes_zero_inflation` `Function` <br>
Naive Bayesian classifier using zero-inflation model to deal with continuous variables.<br>
Parameter: `lamda` Laplace smoothing paramters<br>
* `eva_tan_bayes_discretization` `Function` <br>
TAN classifier with continuous variable discretization.<br>
Parameter: `lamda` Laplace smoothing paramters<br>
* `eva_tan_bayes_zero_inflation` `Function` <br>
TAN classifier using zero-inflation model to deal with continuous variables.<br>
Parameter: `lamda` Laplace smoothing paramters<br>
* `eva_NBTree` `Funciton` <br>
NBTree classifier<br>
Parameter: `lamda` Laplace smoothing paramters<br>
* `eva_adaboost_nb` `Function` <br>
Naive-Bayes based on Adaboost.<br>
Parameter: `lamda` Laplace smoothing paramters<br>
`m` iteration times<br>

### Result
|Classifier|Accuracy|Precision|Recall|
|---|---|---|---|
|Naive Bayes|82.39%|85.36%|92.53%|
|TAN Bayes|82.74%|89.30%|87.62%|
|NBTree|82.86%|92.91%|83.67%|
|adaboost-NB|84.01%|90.42%|88.14%|


****

|Author|RongRongJi|
|---|---
|Contact|[homepage](https://github.com/RongRongJi)