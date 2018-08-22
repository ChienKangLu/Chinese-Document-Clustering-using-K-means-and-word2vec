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

## Word2vec
+ Using the Skip-gram model: use a word to predict a target context
+ 

