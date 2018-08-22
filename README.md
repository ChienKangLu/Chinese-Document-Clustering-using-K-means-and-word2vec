# Chinese-Document-Clustering-using-K-means-and-word2vec
Document Clustering is a method for finding structure within a collection of documents, so that similar documents can be grouped into categories. This is an Unsupervised grouping of text documents into meaningful groups, usually representing topics in the document collection

## Difficulties
+ Unstructured data: Document is composed by words 
+ How to capture the semantic in the document
+ How to encode the document: Let computer can process them
+ Chinese word is different from English: Maybe two or three words concatenate together be the smallest semantic unit

## Experiment Flow chart
<p align="center">
  <img width="400" src="https://github.com/ChienKangLu/Chinese-Document-Clustering-using-K-means-and-word2vec/blob/master/flow-chart.png" />
</p>

## Raw data preprocessing
1. Use Foxpro to save documents
2. Split documents into sentences
3. Remove alpha and number, only preserve chinese 
4. Use jeibar to do sentense segmentation

## Word2vec(genism,python 3.6)
+ Using the Skip-gram model: use a word to predict a target context
+ The method is a single hidden layer neural network 
+ The Input is one-hot word vector
+ The output is the corresponding word around the input word depend on the window size 
+ The first trained parameter matrix  is the dictionary we want(every word will get a vector to represent themselves)
+ Dimension: 250
+ Window size: 5
+ Get a word dictionary which contains 23,767 vectors from 2,999,179 tokens 

## Document vector
+ I get the document vector by taking the average of all word vector in a document
+ Use PCA to reduce the dimension

## Clustering 
+ K means
+ Use silhouette analysis to find how many cluster number is better

## Document visualization
![image](https://github.com/ChienKangLu/Chinese-Document-Clustering-using-K-means-and-word2vec/blob/master/documents.png)
![image](https://github.com/ChienKangLu/Chinese-Document-Clustering-using-K-means-and-word2vec/blob/master/document%20clusters.png)




