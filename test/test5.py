# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import csv
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
#計算mean document vector
def saveTofile(output,doc_id,size,vector):
    output.write(str(doc_id)+","+str(size))
    for n in vector:
        output.write(","+str(n))
    output.write("\n")

def add(a,b):
    a_np=np.array(a)
    b_np=np.array(b)
    return a_np+b_np
def main():
    print("good")
    '''
    a=[1,2]
    b=[3,7]
    print(add(a,b))
    sum=[0,0]
    a=[1,2]
    b=[3,7]
    sum=add(sum,a)
    sum=add(sum,b)
    print(sum)
    print(sum/2)
    a=[0]*250
    '''
    

    #dictionary
    word_vectors = KeyedVectors.load("D:\\NTUST\\人工智慧\\final\\csv\\code250.model.bin")
    '''
    print("無法",word_vectors["無法"])
    print("市長",word_vectors["市長"])
    print("無法+市長",add(word_vectors["無法"],word_vectors["市長"]))
    #-5.60508668e-01  -0.56
    #1.70959115e-01    0.17
    #-0.38954955
    #print(add(add(word_vectors["無法"],word_vectors["市長"]),[1]*250))
    '''

    '''
    wordvec="D:\\NTUST\\人工智慧\\final\\csv\\wordvec.csv"
    wordvec_file = open(wordvec, 'r',encoding='utf-8')
    wordvec_filecsvCursor = csv.reader(wordvec_file)
    for row in wordvec_filecsvCursor:
        print(row[0]+","+row[1])
        id_list.append([int(row[0]),int(row[1])])
        break
    '''

    #取出doc_id,sent_id
    id_list=[]
    id="D:\\NTUST\\人工智慧\\final\\csv\\sentence_retmoveAlphanum.csv"
    id_file = open(id, 'r',encoding='utf-8')
    id_filecsvCursor = csv.reader(id_file)
    next(id_filecsvCursor, None)  # skip the headers
    for row in id_filecsvCursor:
        #print(row[0]+"~~"+row[1])
        id_list.append([int(row[0]),int(row[1])])


    texts_num = 0
    sentence="D:\\NTUST\\人工智慧\\final\\csv\\sentence_retmoveAlphanum_seg.txt"

    output = open('D:\\NTUST\\人工智慧\\final\\csv\\document_vector.csv', 'w')


    with open(sentence, 'r' , encoding='utf8') as content:
        sentenceSum=[0]*250
        word_size=0
        doc=id_list[0][0]#doc_id
        for line in content:
            #print(id_list[texts_num])#doc_id,sent_id
            now_doc=id_list[texts_num][0]#doc_id
            if(now_doc!= doc):
                sentenceSum=sentenceSum/word_size
                print(str(doc)+"~",end='')
                print(word_size)
                #print(sentenceSum)
                saveTofile(output,doc,word_size,sentenceSum)
                sentenceSum=[0]*250
                word_size = 0
                doc=now_doc
                #break
            #print(line)
            words=line.split(" ")
            del words[-1]
            for word in words:
                #print(word)
                if(word in word_vectors):
                    word_size += 1
                    sentenceSum=add(sentenceSum,word_vectors[word])
                    #print(word_vectors[word])
            #print(len(words))
            #print(sentenceSum)
            texts_num+=1
    sentenceSum = sentenceSum / word_size
    print(str(doc) + "~", end='')
    print(word_size)
    saveTofile(output, doc, word_size, sentenceSum)
    output.close()


if __name__ == "__main__":
    main()