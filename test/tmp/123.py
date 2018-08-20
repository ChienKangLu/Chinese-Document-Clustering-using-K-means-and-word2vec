# -*- coding: utf-8 -*-
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import sys

from gensim.corpora import WikiCorpus
import jieba
from opencc import OpenCC

import csv
import os

import numpy as np
def main():
    print("good")

    #jieba 測試
    #seg_list1 = jieba.cut("我上高中的时候，学校查迟到的，抓到后就罚班主任的钱，结果那天我迟到被门口警卫拦住，问我叫什么，我说我叫田某某（我同桌名字），门卫记上后说会通知我们班主任，然后就让我进去了，进教室一看，我同桌没来呢，大概五分钟左右，他进来了，我没好意思看他，就低头看小说，十分钟后，我们班主任把我们叫出去，说学校通知他我们俩迟到，罚去操场跑步5圈，我就纳闷了，我没写我名字啊，然后我看田某，他也用同样的眼光看我，然后我们异口同声的说，次奥！！！", cut_all=False)
    #print("Default Mode: " + "~".join(seg_list1))  # 精确模式
    str="wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww"
    result = re.findall("(.{1,25})(\\s|$)", str)

    print (result)
    a="asd"
    b="ert"
    print(a+b)
    # 輸出 Default Mode: 我/ 来到/ 北京/ 清华大学

    d=[[1,2],[3,4]]
    countid = 0;
    centroids_id = []
    for coordinate in d:
        if (coordinate[0] == 1 and coordinate[1] == 2):
            centroids_id.append(countid)
        countid += 1;
    print(centroids_id)

    x=np.array([3,0])
    y=np.array([0,4])
    print(x)
    print(y)
    print(x+y)
    print(np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)))

if __name__ == "__main__":
    main()


