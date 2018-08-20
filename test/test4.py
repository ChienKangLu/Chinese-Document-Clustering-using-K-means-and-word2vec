# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import models

from gensim.models.keyedvectors import KeyedVectors

import csv

#使用word2vec訓練完的向量,輸出dictionary(wordvec.csv)
def main():
    print("good")
    model = models.Word2Vec.load("D:\\NTUST\\人工智慧\\final\\csv\\code250.model.bin")
    # print(len(model.wv.vocab))
    #print(model.wv.vocab[0])
    word_vectors = KeyedVectors.load("D:\\NTUST\\人工智慧\\final\\csv\\code250.model.bin")
    print(word_vectors["市長"])
    '''
    output = open('D:\\NTUST\\人工智慧\\final\\csv\\wordvec.csv', 'w')
    count=0;
    for word in model.wv.vocab:
        print(word+"~~", end='')
        output.write(word)
        print(len(word_vectors[word]),"~", end='')
        for item in word_vectors[word]:
            print(item, end='')
            output.write(",")
            output.write(str(item))



        output.write("\n")
        print()
        #count+=1
        #if(count==20):
        #    break
    output.close()
    '''
    '''
    res = model.most_similar("台灣",topn = 100)
    for item in res:
        print(item[0] + "," + str(item[1]))
    '''

if __name__ == "__main__":
    main()
