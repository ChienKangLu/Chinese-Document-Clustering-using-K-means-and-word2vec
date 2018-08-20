# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import sys

from gensim.corpora import WikiCorpus
import jieba
from opencc import OpenCC

import csv
import os
def main():
    print("good")

    '''
    #jieba 測試
    seg_list1 = jieba.cut("五都市長25日將就職，由於縣市合併，一口氣將有五位現任縣市長「畢業」，他們何去何從甚受關注，其中三人卸任後的規畫都與中國大陸有關，包括綠營出身的高雄縣長楊秋興與台南縣長蘇煥智，顯示兩岸熱潮讓人無法輕視。", cut_all=False)
    print("Default Mode: " + "~".join(seg_list1))  # 精确模式

    seg_list2 = jieba.cut("五都市長日將就職 由於縣市合併 一口氣將有五位現任縣市長 畢業  他們何去何從甚受關注 其中三人卸任後的規畫都與中國大陸有關 包括綠營出身的高雄縣長楊秋興與台南縣長蘇煥智 顯示兩岸熱潮讓人無法輕視 ", cut_all=False)
    print("Default Mode: " + "~".join(seg_list2))  # 精确模式

    # 輸出 Default Mode: 我/ 来到/ 北京/ 清华大学
    '''


    '''
    #opencc 測試(中文簡繁轉換)
    openCC = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese
    # can also set conversion by calling set_conversion
    # openCC.set_conversion('s2tw')
    to_convert = '陸建綱'
    converted = openCC.convert(to_convert)
    print(converted)
    '''

    '''
    #直接讀取檔案
    texts_num = 0
    with open(fileName, 'r') as content:
        for line in content:
            print(line)
            texts_num+=1
            if texts_num >= 10 :
                break
    '''



    #讀取mark.csv
    markFileName="D:\\NTUST\\人工智慧\\final\\csv\\mark.csv"
    markfile = open(markFileName, 'r',encoding='utf-8')
    markcsvCursor = csv.reader(markfile)
    next(markcsvCursor, None)  # skip the headers
    markset=set() #無序不重複
    for row in markcsvCursor:
        print(row[3]+"->",ord(row[3]))
        markset.add(row[3])
    print("markSize: ",len(markset))
    markfile.close()


    #sentence segmentation
    #從csv讀取檔案
    fileName="D:\\NTUST\\人工智慧\\final\\csv\\sentence_retmoveAlphanum.csv"
    file = open(fileName, 'r',encoding='utf-8')
    filecsvCursor = csv.reader(file)
    next(filecsvCursor, None)  # skip the headers

    output = open('D:\\NTUST\\人工智慧\\final\\csv\\sentence_retmoveAlphanum_seg.txt','w')

    texts_num = 0
    for row in filecsvCursor:
            nowSentense=row[2]
            #print(nowSentense)
            cutSentence=jieba.cut(nowSentense,cut_all=False)
            tempString=""
            for word in cutSentence:
                if(word not in markset):
                    tempString+=word+" "
                    #print(word+" ")
                    output.write(word+" ")
            output.write("\n")
            #print(tempString)
            texts_num+=1
            if texts_num % 10 == 0:
                print("已完成前 %d 行的斷詞" % texts_num)
            #if(texts_num==3):
            #    break

    file.close()
    output.close()


if __name__ == "__main__":
    main()
