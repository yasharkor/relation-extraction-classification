
# Relation Extraction & Classification
## Installation and Execution Instructions
- Clone the repository
- In your python environment ensure that you have installed the sklearn package, it can be installed with the following commands:
  ```python3 -m pip install scikit-learn==0.20 ``` (version 0.20 only needed on lab machines with python 3.5)
  - Navigate to the code folder of the repository (```assignment-5---bow-relation-classification-HonestPretzels\code``` on github)
  - from the terminal execute the following command: ``` python3 .\train.py --train [path to train file] --test [path to test file] --dev [choices=['True', 'False']] --output [path to output file]  ```
  - The arguments can be entered in any order
  - The **[path to train file]** argument is the relative path to the file you wish train NB classifier
   - The **[path to test file]** argument is the relative path to the file you wish predict. It can be either test file or eval file
   - The **[path to output file]** argument is the relative path to the file you wish to write the predicted classes to
   - The **[choices=['True', 'False']]** argument is the mode you wish to run the program in.

     - **False** - test file have original_label. Default mode
     - **True** - test file does not includes the original_label and it is blank (eval file)
 

    - An example usage of this program would be: ``` python3 ./train.py --train ../data/train.csv --test ../data/test.csv --dev 'False' --output ../output/output_test.csv```
    - The above command will train the classifier and will test it on the test data

    - An example usage of this program would be: ``` python3 ./train.py  --train ../data/train.csv --test ../data/eval.csv --dev 'True' --output ../output/output_eval.csv ```
   - The above command will train the classifier and will test it on the eval data that does not have original_labels. Does not provide any accuracy or other metrics on the test file.
## Data

The training data can be found in [data/train.csv](data/train.csv), the labelled test data can be found in [data/test.csv](data/test.csv), and the unlablled eval data can be found in [data/eval.csv](data/eval.csv).

## Resources Consulted
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- [Text book chapter 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

## Introduction
Relation extraction is the task of “finding and classifying semantic relations among the text entities”. Given a sentence containing two entities (called head and tail), the goal is to classify the relation between the head entity and the tail. For example, from the sentence “Newton served as the president of the Royal Society”, the relation “is a member of” between the head entity “Newton” and the tail entity “the Royal Society” can be extracted.

In this project, we will build a Naive Bayes model based on bag-of-word (BoW) features to classify the relation of a sentence. The program process and classify a sentence as indicating one of the following relations: publisher, director, performer, and characters.

## Input
Three csv files: train.csv, eval.csv, and test.csv, adapted from the [FewRel](https://www.aclweb.org/anthology/D18-1514) dataset, is provided.


## Part 1: Classify Text
We Write and document a program that trains a Naive Bayes classifier for the 4-class classification problem at hand. The program will process texts and classify them as belonging to one of the following classes: publisher, director, performer, and characters. The NB classifier is implemented from scratch, instead of calling libraries like scikit-learn.

The program take train.csv as input, train the model, and print the training accuracy (using 3-fold cross validation). After training the model, the program  take dev.csv and test.csv as input, write the predictions to files, and print the accuracy on the test set.

## Suggestions

[Chapter 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf) of the textbook provides detailed instructions for building a NB text classifier, as well as the methods for evaluation. 
[Chapter 18](https://web.stanford.edu/~jurafsky/slp3/18.pdf) describes the relation extraction task, which you can read if you are interested.

## Further reading
The method for relation extraction described here, i.e., training a text classifier to classify the relations, requires a substantial amount of training data for each of the relations. However, the method is not very scalable as it is not easy to obtain a large amount of labelled data to support supervised learning. To alleviate this issue, some clever ideas emerged. [Distant supervision](https://web.stanford.edu/~jurafsky/mintz.pdf) could generate a “silver-standard” training set by first finding many pairs of entities that belong to the relation of interest, and then using all sentences containing those entities as training data for that relation. [Few-shot learning](https://www.aclweb.org/anthology/D18-1514.pdf) tries to train ML models that could easily generalize to unseen classes by only looking at a few examples from the unseen class.  

## Acknowledgement


This is a solution to Assignment 5 for CMPUT 497 - Intro to NLP at the University of Alberta, created during the Fall 2020 semester. Thomas Maurer: tmaurer@ualberta.ca, Yashar Kor: yashar@ualberta.ca
